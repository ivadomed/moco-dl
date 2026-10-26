"""
Inference: apply a trained rigid (rotation + translation) registration model
to motion-correct a 4D spinal cord fMRI NIfTI volume, slicewise.

Reference strategy (choose with --reference):
- 'average' (default): each slice's reference is the mean image across all its
  timepoints. Every frame registers independently against this single stable
  target -- no sequential dependency, so no drift accumulation across the run.
- 'first': each slice's reference is simply its first timepoint (t=0).

Either way, every timepoint (including t=0) is warped the same, consistent way
against the chosen reference -- there is no special-cased "anchor" frame left
untouched, unlike the earlier sequential approach.

Because the model predicts a single rigid transform (not a dense field), the
same predicted affine matrix is applied directly to the *raw*-intensity slice
via grid_sample, preserving original intensity scale in the output.

Usage:
    python infer.py \
        --input sub-01_bold.nii.gz \
        --checkpoint checkpoints/best_model.pt \
        --output sub-01_bold_corrected.nii.gz \
        --reference average \
        --interpolation bilinear

Requires: pip install monai nibabel torch
"""

import argparse

import numpy as np
import nibabel as nib
import torch

from rigid_model import RigidRegistrationNet, warp_with_affine


# --------------------------------------------------------------------------
# Config -- MUST match the architecture/preprocessing used in train.py
# --------------------------------------------------------------------------
TARGET_SHAPE = (128, 128)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    operation exactly via from_model_space.
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


def from_model_space(slice_target: np.ndarray, params: dict) -> np.ndarray:
    """Exact inverse of to_model_space: undo the crop, then undo the pad."""
    ph, pw = params["padded_shape"]
    start_h, start_w = params["start_h"], params["start_w"]
    th, tw = slice_target.shape

    canvas = np.zeros((ph, pw), dtype=slice_target.dtype)
    canvas[start_h:start_h + th, start_w:start_w + tw] = slice_target

    pad_top, pad_left = params["pad_top"], params["pad_left"]
    oh, ow = params["orig_shape"]
    return canvas[pad_top:pad_top + oh, pad_left:pad_left + ow]


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


def run_inference(input_path: str, checkpoint_path: str, output_path: str,
                   reference: str = "average", interpolation: str = "bilinear"):
    img = nib.load(input_path)
    data = img.get_fdata(dtype=np.float32)  # (X, Y, Z, T)
    X, Y, Z, T = data.shape
    print(f"Loaded {input_path}: shape {data.shape}, reference={reference}, interpolation={interpolation}")

    model = build_model()
    model = load_checkpoint(model, checkpoint_path)

    reference_raw = compute_reference(data, reference)  # (X, Y, Z), fixed for the whole run

    # pad/crop params only depend on the in-plane (X, Y) shape, which is the
    # same for every slice and timepoint -- compute once and reuse.
    _, params = to_model_space(data[:, :, 0, 0])

    corrected = np.zeros_like(data)

    for t in range(T):
        moving_raw = data[:, :, :, t]  # (X, Y, Z)

        moving_norm_batch, fixed_norm_batch, moving_raw_padded_batch = [], [], []
        for z in range(Z):
            m_raw = moving_raw[:, :, z]
            f_raw = reference_raw[:, :, z]

            m_norm_padded, _ = to_model_space(normalize(m_raw))
            f_norm_padded, _ = to_model_space(normalize(f_raw))
            m_raw_padded, _ = to_model_space(m_raw)

            moving_norm_batch.append(m_norm_padded)
            fixed_norm_batch.append(f_norm_padded)
            moving_raw_padded_batch.append(m_raw_padded)

        moving_norm_t = torch.from_numpy(np.stack(moving_norm_batch)[:, None]).float().to(DEVICE)
        fixed_norm_t = torch.from_numpy(np.stack(fixed_norm_batch)[:, None]).float().to(DEVICE)
        moving_raw_t = torch.from_numpy(np.stack(moving_raw_padded_batch)[:, None]).float().to(DEVICE)

        with torch.no_grad():
            _, affine_matrix = model(moving_norm_t, fixed_norm_t)
            # apply the predicted rigid transform to the RAW (un-normalized) slice
            # so the saved output keeps original intensity scale; interpolation
            # mode is configurable here since it doesn't need to be differentiable
            # at inference time (unlike during training, where bilinear is required).
            warped_raw = warp_with_affine(moving_raw_t, affine_matrix, mode=interpolation)

        warped_raw_np = warped_raw.squeeze(1).cpu().numpy()  # (Z, th, tw)

        for z in range(Z):
            corrected[:, :, z, t] = from_model_space(warped_raw_np[z], params)

        print(f"  timepoint {t + 1}/{T} corrected")

    out_img = nib.Nifti1Image(corrected.astype(np.float32), affine=img.affine, header=img.header)
    nib.save(out_img, output_path)
    print(f"Saved motion-corrected volume to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Rigid motion-correct a 4D fMRI NIfTI with a trained model.")
    parser.add_argument("--input", required=True, help="Path to the input 4D NIfTI (.nii/.nii.gz)")
    parser.add_argument("--checkpoint", required=True, help="Path to the trained model checkpoint (.pt)")
    parser.add_argument("--output", required=True, help="Path to write the corrected 4D NIfTI")
    parser.add_argument("--reference", choices=["average", "first"], default="average",
                         help="Reference strategy: mean across timepoints (default) or the first timepoint")
    parser.add_argument("--interpolation", choices=["bilinear", "nearest"], default="bilinear",
                         help="Warp interpolation mode. 'nearest' gives sharper but blockier output.")
    args = parser.parse_args()

    run_inference(args.input, args.checkpoint, args.output,
                  reference=args.reference, interpolation=args.interpolation)


if __name__ == "__main__":
    main()
