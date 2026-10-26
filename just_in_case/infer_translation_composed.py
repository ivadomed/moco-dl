"""
Makes a double inference with both models in a SINGLE interpolation step.

Inference: apply two trained rigid (rotation + translation) registration models
to motion-correct a 4D spinal cord fMRI NIfTI volume, slicewise.

Pipeline (this is the key design point of this script):
1. Take the RAW (uncropped) 4D NIfTI as input.
2. Run the external spinal-cord crop tool (`sc_crop`) to produce a cropped version.
3. Run Model 1 on the CROPPED data, getting theta1.
4. Warp the CROPPED data with theta1 to get an intermediate corrected crop.
5. Run Model 2 on the INTERMEDIATE data, getting theta2.
6. Compose the matrices (theta2 * theta1) to get a single rigid transformation.
7. Transfer that combined transform from the crop's coordinate space into the RAW
   image's coordinate space (coord_transform.transfer_theta_to_raw), and warp
   the RAW slice directly ONCE -- minimizing interpolation blurring.
8. Save the corrected output at the RAW image's full resolution/affine/header.
"""

import argparse
import os
import subprocess
import tempfile

import numpy as np
import nibabel as nib
import torch

from rigid_model_translation import RigidRegistrationNet, warp_with_affine
from coord_transform import transfer_theta_to_raw


# --------------------------------------------------------------------------
# Config -- MUST match the architecture/preprocessing used in train.py
# --------------------------------------------------------------------------
TARGET_SHAPE = (64, 64)
DEVICE = "cpu"  # torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
def build_model():
    return RigidRegistrationNet().to(DEVICE)


def load_checkpoint(model, checkpoint_path):
    state_dict = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    return model


# --------------------------------------------------------------------------
# Spinal cord crop (external tool)
# --------------------------------------------------------------------------
def run_sc_crop(raw_path: str, cropped_path: str):
    cmd = ["sc_crop", "-i", raw_path, "-o", cropped_path]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    if not os.path.exists(cropped_path):
        raise FileNotFoundError(
            f"sc_crop did not produce the expected output at {cropped_path} -- "
            f"check whether your sc_crop uses different flags or a fixed output naming convention."
        )


# --------------------------------------------------------------------------
# Preprocessing (same normalization as training) + invertible pad/crop
# --------------------------------------------------------------------------
def normalize(slice_2d: np.ndarray) -> np.ndarray:
    slice_2d = slice_2d.astype(np.float32)
    p1, p99 = np.percentile(slice_2d, (1, 99))
    slice_2d = np.clip(slice_2d, p1, p99)
    denom = (p99 - p1) if (p99 - p1) > 1e-8 else 1e-8
    return (slice_2d - p1) / denom


def to_model_space(slice_2d: np.ndarray, target_shape=TARGET_SHAPE):
    slice_2d = np.asarray(slice_2d)
    orig_shape = slice_2d.shape
    h, w = orig_shape
    th, tw = target_shape

    pad_h, pad_w = max(th - h, 0), max(tw - w, 0)
    pad_top, pad_bottom = pad_h // 2, pad_h - pad_h // 2
    pad_left, pad_right = pad_w // 2, pad_w - pad_w // 2

    padded = np.pad(slice_2d, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="constant")
    ph, pw = padded.shape

    start_h = max((ph - th) // 2, 0)
    start_w = max((pw - tw) // 2, 0)
    cropped = padded[start_h:start_h + th, start_w:start_w + tw]

    params = {
        "orig_shape": orig_shape,
        "padded_shape": (ph, pw),
        "pad_top": pad_top,
        "pad_left": pad_left,
        "start_h": start_h,
        "start_w": start_w,
    }
    return cropped, params


# --------------------------------------------------------------------------
# Crop <-> raw coordinate correspondence
# --------------------------------------------------------------------------
def get_crop_offset_voxels(raw_img: nib.Nifti1Image, cropped_img: nib.Nifti1Image) -> np.ndarray:
    world_origin = cropped_img.affine @ np.array([0, 0, 0, 1])
    raw_voxel_origin = np.linalg.inv(raw_img.affine) @ world_origin
    offset = raw_voxel_origin[:3]
    rounded = np.round(offset)
    if not np.allclose(offset, rounded, atol=1e-2):
        print(f"WARNING: crop offset {offset} is not close to integer voxel indices -- "
              f"the crop may involve resampling/rotation, which this script does not handle correctly.")
    return rounded.astype(int)


# --------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------
def compute_reference(data: np.ndarray, reference: str) -> np.ndarray:
    if reference == "average":
        return data.mean(axis=-1)
    elif reference == "first":
        return data[:, :, :, 0].copy()
    else:
        raise ValueError(f"Unknown reference strategy: {reference!r}")


def run_inference(raw_input_path: str, checkpoint1_path: str, checkpoint2_path: str, output_path: str,
                   reference: str = "average", interpolation: str = "bilinear",
                   cropped_path: str = None, keep_cropped: bool = False):
    
    # --- 1. Crop the raw image for the model ---
    cleanup_cropped = False
    if cropped_path is None:
        cropped_path = tempfile.mktemp(suffix="_sc_crop.nii.gz")
        cleanup_cropped = not keep_cropped
    run_sc_crop(raw_input_path, cropped_path)

    raw_img = nib.load(raw_input_path)
    raw_data = raw_img.get_fdata(dtype=np.float32)  
    cropped_img = nib.load(cropped_path)
    cropped_data = cropped_img.get_fdata(dtype=np.float32)  

    X_raw, Y_raw, Z_raw, T_raw = raw_data.shape
    Xc, Yc, Zc, Tc = cropped_data.shape
    if Tc != T_raw:
        raise ValueError(f"Timepoint count mismatch between raw ({T_raw}) and cropped ({Tc})")

    print(f"Raw shape: {raw_data.shape} | Cropped shape: {cropped_data.shape} "
          f"| reference={reference}, interpolation={interpolation}")

    # --- 2. Determine where the crop sits within the raw volume ---
    axis0_offset, axis1_offset, axis2_offset = get_crop_offset_voxels(raw_img, cropped_img)
    print(f"Detected crop offset within raw volume (X, Y, Z axes): "
          f"({axis0_offset}, {axis1_offset}, {axis2_offset})")

    crop_center_in_raw_px = (axis1_offset + Yc / 2.0, axis0_offset + Xc / 2.0)
    oz = axis2_offset

    # --- 3. Setup Models & Reference ---
    model1 = build_model()
    model1 = load_checkpoint(model1, checkpoint1_path)
    
    model2 = build_model()
    model2 = load_checkpoint(model2, checkpoint2_path)
    
    reference_cropped = compute_reference(cropped_data, reference)

    # --- 4. Correction ---
    corrected = raw_data.copy()

    for t in range(T_raw):
        moving_norm_batch, fixed_norm_batch = [], []
        for z in range(Zc):
            m_raw = cropped_data[:, :, z, t]
            f_raw = reference_cropped[:, :, z]
            m_norm_padded, _ = to_model_space(normalize(m_raw))
            f_norm_padded, _ = to_model_space(normalize(f_raw))
            moving_norm_batch.append(m_norm_padded)
            fixed_norm_batch.append(f_norm_padded)

        moving_norm_t = torch.from_numpy(np.stack(moving_norm_batch)[:, None]).float().to(DEVICE)
        fixed_norm_t = torch.from_numpy(np.stack(fixed_norm_batch)[:, None]).float().to(DEVICE)

        with torch.no_grad():
            # Première Inférence (Model 1)
            _, affine_matrix1 = model1(moving_norm_t, fixed_norm_t)
            
            # Création de l'image intermédiaire (déformée par Model 1) pour l'estimer dans le Model 2
            # On utilise "bilinear" pour cette étape interne afin que le réseau voie une image fluide
            moving_corrected_t = warp_with_affine(moving_norm_t, affine_matrix1, mode="bilinear")

            # Seconde Inférence (Model 2) sur l'image intermédiaire
            _, affine_matrix2 = model2(moving_corrected_t, fixed_norm_t)

        affine1_np = affine_matrix1.cpu().numpy()
        affine2_np = affine_matrix2.cpu().numpy()

        for z in range(Zc):
            raw_z = z + oz
            if raw_z < 0 or raw_z >= Z_raw:
                continue 

            # Combiner mathématiquement les matrices 2x3 en les passant en 3x3
            A1 = np.vstack([affine1_np[z], [0.0, 0.0, 1.0]])
            A2 = np.vstack([affine2_np[z], [0.0, 0.0, 1.0]])
            
            # CORRECTION : En mapping inverse (PyTorch grid_sample), 
            # la composition T1 puis T2 s'écrit T1 @ T2.
            A_combined = A1 @ A2
            
            affine_combined_2x3 = A_combined[:2, :] # Retour à la dimension 2x3

            # Transférer la matrice COMBINÉE de l'espace du modèle à l'espace raw
            theta_raw = transfer_theta_to_raw(
                affine_combined_2x3, TARGET_SHAPE, crop_center_in_raw_px, (X_raw, Y_raw)
            )
            theta_raw_t = torch.from_numpy(theta_raw).unsqueeze(0).float().to(DEVICE)

            raw_slice_t = torch.from_numpy(raw_data[:, :, raw_z, t]).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
            
            # Appliquer l'unique déformation combinée sur l'image raw
            warped_raw_slice = warp_with_affine(raw_slice_t, theta_raw_t, mode=interpolation)
            corrected[:, :, raw_z, t] = warped_raw_slice[0, 0].cpu().numpy()

        print(f"  timepoint {t + 1}/{T_raw} corrected (single combined step)")

    out_img = nib.Nifti1Image(corrected.astype(np.float32), affine=raw_img.affine, header=raw_img.header)
    nib.save(out_img, output_path)
    print(f"Saved motion-corrected volume (raw resolution/format) to {output_path}")

    if cleanup_cropped and os.path.exists(cropped_path):
        os.remove(cropped_path)


def main():
    parser = argparse.ArgumentParser(
        description="Rigid motion-correct a raw 4D fMRI NIfTI: apply two model corrections in a single interpolation step."
    )
    parser.add_argument("--input", required=True, help="Path to the RAW (uncropped) input 4D NIfTI (.nii/.nii.gz)")
    parser.add_argument("--checkpoint1", required=True, help="Path to the first trained model checkpoint (.pt)")
    parser.add_argument("--checkpoint2", required=True, help="Path to the second trained model checkpoint (.pt)")
    parser.add_argument("--output", required=True, help="Path to write the corrected 4D NIfTI")
    parser.add_argument("--reference", choices=["average", "first"], default="average",
                         help="Reference strategy")
    parser.add_argument("--interpolation", choices=["bilinear", "nearest"], default="bilinear",
                         help="Warp interpolation mode, applied to the raw image.")
    parser.add_argument("--cropped-path", default=None,
                         help="Optional path for the intermediate cropped file")
    parser.add_argument("--keep-cropped", action="store_true",
                         help="Keep the intermediate cropped file")
    args = parser.parse_args()

    # Appel de l'inférence unique avec les DEUX checkpoints
    run_inference(args.input, args.checkpoint1, args.checkpoint2, args.output,
                  reference=args.reference, interpolation=args.interpolation,
                  cropped_path=args.cropped_path, keep_cropped=args.keep_cropped)


if __name__ == "__main__":
    main()