"""
Makes a double inference with both models 


Inference: apply a trained rigid (rotation + translation) registration model
to motion-correct a 4D spinal cord fMRI NIfTI volume, slicewise.

Pipeline (this is the key design point of this script):
1. Take the RAW (uncropped) 4D NIfTI as input.
2. Run the external spinal-cord crop tool (`sc_crop`) to produce a cropped
   version -- this matches the distribution the model was trained on.
3. Run the model on the CROPPED data, per slice/timepoint, exactly as before
   -- this gives a predicted rigid transform (theta) for every (z, t).
4. Transfer that transform from the crop's coordinate space into the RAW
   image's coordinate space (coord_transform.transfer_theta_to_raw), and warp
   the RAW slice directly -- never the cropped one.
5. Save the corrected output at the RAW image's full resolution/affine/header.

Why: applying the correction to the raw image (not the crop) means the tight
crop's zero-padding border is NEVER part of the output, and never re-enters the
model on a later pass. The raw image's full FOV comfortably contains real
anatomical displacement, so no black/asymmetric-padding artifacts are introduced.

Reference strategy (choose with --reference), computed from the CROPPED data:
- 'average' (default): each slice's reference is the mean image across all its
  timepoints. Every frame registers independently against this single stable
  target -- no sequential dependency, so no drift accumulation across the run.
- 'first': each slice's reference is simply its first timepoint (t=0).

Usage:
    python infer_2d.py \
        --input raw_sub-01_bold.nii.gz \
        --checkpoint1 checkpoints/first_model.pt \
        --checkpoint2 checkpoints/second_model.pt \
        --output sub-01_bold_corrected.nii.gz \
        --reference average \
        --interpolation bilinear

Requires: pip install monai nibabel torch
Requires the `sc_crop` command to be available on PATH. NOTE: the exact CLI
flags below (`--input`/`--output`) are a best guess based on the flag names
you gave -- adjust run_sc_crop() if your tool's interface differs.
"""

import argparse
import os
import subprocess
import tempfile

import numpy as np
import nibabel as nib
import torch

from rigid_model import RigidRegistrationNet, warp_with_affine
from coord_transform import transfer_theta_to_raw


# --------------------------------------------------------------------------
# Config -- MUST match the architecture/preprocessing used in train.py
# --------------------------------------------------------------------------
TARGET_SHAPE = (64, 64)                # MUST match train.py's TARGET_SHAPE exactly
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
    """
    Runs the external sc_crop tool to crop the raw image around the spinal cord.
    NOTE: verify/adjust these flags against your actual sc_crop CLI -- this is
    the interface you described (--input / --output).
    """
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
    """Percentile-clipped min-max normalization, robust to outlier voxels."""
    slice_2d = slice_2d.astype(np.float32)
    p1, p99 = np.percentile(slice_2d, (1, 99))
    slice_2d = np.clip(slice_2d, p1, p99)
    denom = (p99 - p1) if (p99 - p1) > 1e-8 else 1e-8
    return (slice_2d - p1) / denom


def to_model_space(slice_2d: np.ndarray, target_shape=TARGET_SHAPE):
    """
    Center pad (if smaller) and/or center crop (if larger) to target_shape.
    Returns the transformed slice plus the params needed to invert the
    operation exactly (not needed for warping here, but kept for parity/debugging).
    """
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
    """
    Returns the (x, y, z) voxel-index offset of the cropped image's voxel (0,0,0)
    within the raw image's voxel grid, using the two images' NIfTI affines.
    Assumes the crop is a pure axis-aligned translation (no resampling/rotation),
    which holds for a standard ROI crop that preserves voxel spacing/orientation.
    """
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
    """data is (X, Y, Z, T). Returns (X, Y, Z) reference volume."""
    if reference == "average":
        return data.mean(axis=-1)
    elif reference == "first":
        return data[:, :, :, 0].copy()
    else:
        raise ValueError(f"Unknown reference strategy: {reference!r}")


def run_inference(raw_input_path: str, checkpoint_path: str, output_path: str,
                   reference: str = "average", interpolation: str = "bilinear",
                   cropped_path: str = None, keep_cropped: bool = False):
    # --- 1. Crop the raw image for the model ---
    cleanup_cropped = False
    if cropped_path is None:
        cropped_path = tempfile.mktemp(suffix="_sc_crop.nii.gz")
        cleanup_cropped = not keep_cropped
    run_sc_crop(raw_input_path, cropped_path)

    raw_img = nib.load(raw_input_path)
    raw_data = raw_img.get_fdata(dtype=np.float32)  # (X, Y, Z, T) -- full FOV
    cropped_img = nib.load(cropped_path)
    cropped_data = cropped_img.get_fdata(dtype=np.float32)  # (Xc, Yc, Zc, T)

    X_raw, Y_raw, Z_raw, T_raw = raw_data.shape
    Xc, Yc, Zc, Tc = cropped_data.shape
    if Tc != T_raw:
        raise ValueError(f"Timepoint count mismatch between raw ({T_raw}) and cropped ({Tc}) -- "
                          f"sc_crop is expected to only crop spatially, not in time.")
    print(f"Raw shape: {raw_data.shape} | Cropped shape: {cropped_data.shape} "
          f"| reference={reference}, interpolation={interpolation}")

    # --- 2. Determine where the crop sits within the raw volume ---
    # get_crop_offset_voxels returns the offset in NATURAL ARRAY AXIS order
    # (axis0, axis1, axis2) -- i.e. (X, Y, Z) in this file's data.shape convention.
    axis0_offset, axis1_offset, axis2_offset = get_crop_offset_voxels(raw_img, cropped_img)
    print(f"Detected crop offset within raw volume (X, Y, Z axes): "
          f"({axis0_offset}, {axis1_offset}, {axis2_offset})")

    # coord_transform.transfer_theta_to_raw expects (x, y) in rigid_model.py's
    # sense: x = width/last-tensor-dim = array axis 1 (this file's "Y"), y =
    # height/second-to-last-tensor-dim = array axis 0 (this file's "X"). This is
    # the OPPOSITE order from (X, Y) as used elsewhere in this file -- swapped
    # deliberately here, do not "fix" this to (X, Y) order.
    crop_center_in_raw_px = (axis1_offset + Yc / 2.0, axis0_offset + Xc / 2.0)  # (x, y)
    oz = axis2_offset

    # --- 3. Model + reference, computed from the CROPPED data (matches training) ---
    model = build_model()
    model = load_checkpoint(model, checkpoint_path)
    reference_cropped = compute_reference(cropped_data, reference)  # (Xc, Yc, Zc)

    # --- 4. Output starts as a copy of the raw data; only the slices covered by
    #     the crop's Z-range get corrected, anything outside is passed through
    #     unchanged since we have no correction information for it.
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
            _, affine_matrix = model(moving_norm_t, fixed_norm_t)  # (Zc, 2, 3)

        affine_matrix_np = affine_matrix.cpu().numpy()

        for z in range(Zc):
            raw_z = z + oz  # this crop slice's index within the raw volume
            if raw_z < 0 or raw_z >= Z_raw:
                continue  # crop's Z-range extends outside the raw volume -- shouldn't happen, skip defensively

            # transfer the predicted transform from model/crop space into raw space
            theta_raw = transfer_theta_to_raw(
                affine_matrix_np[z], TARGET_SHAPE, crop_center_in_raw_px, (X_raw, Y_raw)
            )
            theta_raw_t = torch.from_numpy(theta_raw).unsqueeze(0).float().to(DEVICE)

            raw_slice_t = torch.from_numpy(raw_data[:, :, raw_z, t]).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
            warped_raw_slice = warp_with_affine(raw_slice_t, theta_raw_t, mode=interpolation)
            corrected[:, :, raw_z, t] = warped_raw_slice[0, 0].cpu().numpy()

        print(f"  timepoint {t + 1}/{T_raw} corrected")

    out_img = nib.Nifti1Image(corrected.astype(np.float32), affine=raw_img.affine, header=raw_img.header)
    nib.save(out_img, output_path)
    print(f"Saved motion-corrected volume (raw resolution/format) to {output_path}")

    if cleanup_cropped and os.path.exists(cropped_path):
        os.remove(cropped_path)


def main():
    parser = argparse.ArgumentParser(
        description="Rigid motion-correct a raw 4D fMRI NIfTI: crop for the model, apply the correction to the raw image."
    )
    parser.add_argument("--input", required=True, help="Path to the RAW (uncropped) input 4D NIfTI (.nii/.nii.gz)")
    parser.add_argument("--checkpoint1", required=True, help="Path to the trained model checkpoint (.pt)")
    parser.add_argument("--checkpoint2", required=True, help="Path to the trained model checkpoint (.pt)")
    parser.add_argument("--output", required=True, help="Path to write the corrected 4D NIfTI (raw resolution/format)")
    parser.add_argument("--reference", choices=["average", "first"], default="average",
                         help="Reference strategy (computed from the cropped data): mean across timepoints (default) or the first timepoint")
    parser.add_argument("--interpolation", choices=["bilinear", "nearest"], default="nearest",
                         help="Warp interpolation mode, applied to the raw image. 'nearest' gives sharper but blockier output.")
    parser.add_argument("--cropped-path", default=None,
                         help="Optional path for the intermediate cropped file (default: a temp file, deleted after use unless --keep-cropped)")
    parser.add_argument("--keep-cropped", action="store_true",
                         help="Keep the intermediate cropped file instead of deleting it")
    args = parser.parse_args()

    run_inference(args.input, args.checkpoint1, args.output,
                  reference=args.reference, interpolation="nearest",
                  cropped_path=args.cropped_path, keep_cropped=args.keep_cropped)
    run_inference(args.output, args.checkpoint2, args.output,
                  reference=args.reference, interpolation=args.interpolation,
                  cropped_path=args.cropped_path, keep_cropped=args.keep_cropped)


if __name__ == "__main__":
    main()
