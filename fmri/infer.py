#!/usr/bin/env python
"""
Inference: apply TWO trained rigid registration models in sequence to motion-correct
a 4D spinal cord fMRI NIfTI volume, slicewise.

Pipeline (unchanged from the original infer_adaptive.py):
1. Take the RAW (uncropped) 4D NIfTI as input.
2. Run `sc_crop` to produce a cropped version.
3. STAGE 1: model 1 predicts theta1 on CROPPED data, with a non-regression check:
   if theta1 lowers the slice NCC compared to identity, it is rejected.
4. STAGE 2: model 2 is applied iteratively with early stopping (tol / divergence)
   and per-slice non-regression protection.
5. theta1 and theta2 are COMPOSED, transferred to RAW coordinates, and the RAW
   slice is warped ONCE.
6. The corrected output is saved on the RAW grid (resolution/affine/header).

What changed (speed only, same computations):
- Every network call, non-regression check and warp is BATCHED over all
  (volume, slice) pairs instead of one pair per call (~15,000 tiny torch calls per
  scan before). Batches are split into chunks of BATCH pairs to bound memory.
- The two models are loaded once per process and reused for every scan.
- Per-slice normalisation / padding is vectorised.
- run_inference() returns a dict with the time spent in each step.

Behaviour kept on purpose:
- In stage 2 with reference2="recompute", the ORIGINAL code overwrites the reference
  strategy with "first" (the first volume of the current corrected series), whatever
  `reference` was. This is kept by default so that results match the published runs.
  Pass stage2_reference="average" (CLI: --stage2-reference average) to use the
  temporal mean of the corrected series instead, as described in the paper.

Usage:
    python infer_adaptive.py --input raw_bold.nii.gz --checkpoint1 m1.pt \
        --checkpoint2 m2.pt --output corrected.nii.gz
"""

import argparse
import os
import subprocess
import tempfile
import time

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
BATCH = 4096    # pairs per forward pass / warp call (memory bound only; results do not depend on it)


# --------------------------------------------------------------------------
# Model (loaded once per process)
# --------------------------------------------------------------------------
_MODELS = {}


def build_model():
    return RigidRegistrationNet().to(DEVICE)


def load_checkpoint(model, checkpoint_path):
    state_dict = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def get_model(checkpoint_path):
    """Load a checkpoint once and reuse it (keyed by path + modification time)."""
    st = os.stat(checkpoint_path)
    key = (os.path.realpath(checkpoint_path), st.st_mtime_ns, str(DEVICE))
    if key not in _MODELS:
        _MODELS[key] = load_checkpoint(build_model(), checkpoint_path)
    return _MODELS[key]


# --------------------------------------------------------------------------
# Spinal cord crop (external tool)
# --------------------------------------------------------------------------
def run_sc_crop(raw_path: str, cropped_path: str):
    cmd = ["sc_crop", "-i", raw_path, "-o", cropped_path]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    if not os.path.exists(cropped_path):
        raise FileNotFoundError(f"sc_crop did not produce expected output at {cropped_path}")


# --------------------------------------------------------------------------
# Preprocessing + invertible pad/crop
# --------------------------------------------------------------------------
def normalize(slice_2d: np.ndarray) -> np.ndarray:
    slice_2d = slice_2d.astype(np.float32)
    p1, p99 = np.percentile(slice_2d, (1, 99))
    slice_2d = np.clip(slice_2d, p1, p99)
    denom = (p99 - p1) if (p99 - p1) > 1e-8 else 1e-8
    return (slice_2d - p1) / denom


def normalize_stack(stack: np.ndarray) -> np.ndarray:
    """normalize() applied to every 2D image of stack (..., H, W), vectorised."""
    stack = stack.astype(np.float32)
    lead = stack.shape[:-2]
    flat = stack.reshape(-1, stack.shape[-2] * stack.shape[-1])
    p1, p99 = np.percentile(flat, (1, 99), axis=1)
    p1, p99 = p1.astype(np.float32)[:, None], p99.astype(np.float32)[:, None]
    flat = np.clip(flat, p1, p99)
    denom = np.where((p99 - p1) > 1e-8, p99 - p1, np.float32(1e-8))
    return ((flat - p1) / denom).reshape(lead + stack.shape[-2:])


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


def to_model_space_stack(stack: np.ndarray, target_shape=TARGET_SHAPE) -> np.ndarray:
    """to_model_space() applied to every 2D image of stack (..., H, W), vectorised."""
    h, w = stack.shape[-2:]
    th, tw = target_shape
    pad_h, pad_w = max(th - h, 0), max(tw - w, 0)
    pad = [(0, 0)] * (stack.ndim - 2) + [(pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2)]
    padded = np.pad(stack, pad, mode="constant")
    ph, pw = padded.shape[-2:]
    sh, sw = max((ph - th) // 2, 0), max((pw - tw) // 2, 0)
    return np.ascontiguousarray(padded[..., sh:sh + th, sw:sw + tw])


def get_crop_offset_voxels(raw_img: nib.Nifti1Image, cropped_img: nib.Nifti1Image) -> np.ndarray:
    world_origin = cropped_img.affine @ np.array([0, 0, 0, 1])
    raw_voxel_origin = np.linalg.inv(raw_img.affine) @ world_origin
    offset = raw_voxel_origin[:3]
    return np.round(offset).astype(int)


def compose_thetas(theta_a: np.ndarray, theta_b: np.ndarray) -> np.ndarray:
    a = np.asarray(theta_a, dtype=np.float64)
    b = np.asarray(theta_b, dtype=np.float64)
    bottom = np.broadcast_to(np.array([[0.0, 0.0, 1.0]]), a.shape[:-2] + (1, 3))
    Ma = np.concatenate([a, bottom], axis=-2)
    Mb = np.concatenate([b, bottom], axis=-2)
    return (Ma @ Mb)[..., :2, :].astype(np.float32)


IDENTITY_THETA = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)


# --------------------------------------------------------------------------
# Batched building blocks
# --------------------------------------------------------------------------
def _chunks(n):
    for s in range(0, n, BATCH):
        yield slice(s, min(s + BATCH, n))


def predict_theta(model, moving_np: np.ndarray, fixed_np: np.ndarray) -> np.ndarray:
    """moving_np, fixed_np: (N, H, W) -> theta (N, 2, 3). Chunked forward passes."""
    out = np.empty((moving_np.shape[0], 2, 3), dtype=np.float32)
    with torch.no_grad():
        for c in _chunks(moving_np.shape[0]):
            m = torch.from_numpy(np.ascontiguousarray(moving_np[c, None])).float().to(DEVICE)
            f = torch.from_numpy(np.ascontiguousarray(fixed_np[c, None])).float().to(DEVICE)
            _, affine_matrix = model(m, f)
            out[c] = affine_matrix.cpu().numpy()
    return out


def warp_batch_model_space(moving_np: np.ndarray, theta_np: np.ndarray, mode: str = "bilinear") -> np.ndarray:
    """moving_np (N, H, W), theta_np (N, 2, 3) -> warped (N, H, W). Chunked."""
    out = np.empty(moving_np.shape, dtype=np.float32)
    with torch.no_grad():
        for c in _chunks(moving_np.shape[0]):
            m = torch.from_numpy(np.ascontiguousarray(moving_np[c, None])).float().to(DEVICE)
            th = torch.from_numpy(np.ascontiguousarray(theta_np[c])).float().to(DEVICE)
            out[c] = warp_with_affine(m, th, mode=mode)[:, 0].cpu().numpy()
    return out


def compute_ncc_batch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """compute_ncc() for N pairs of images (N, H, W) -> (N,). Same formula (population std)."""
    a = a.reshape(a.shape[0], -1).astype(np.float32)
    b = b.reshape(b.shape[0], -1).astype(np.float32)
    sa, sb = a.std(axis=1), b.std(axis=1)
    cov = ((a - a.mean(axis=1, keepdims=True)) * (b - b.mean(axis=1, keepdims=True))).mean(axis=1)
    ok = (sa >= 1e-8) & (sb >= 1e-8)
    return np.where(ok, cov / np.where(ok, sa * sb, 1.0), 0.0)


def non_regression_filter_batch(moving: np.ndarray, fixed: np.ndarray, theta_pred: np.ndarray,
                                tolerance: float = 0.0) -> np.ndarray:
    """Keep theta_pred where it does not lower the NCC by more than `tolerance`, else identity."""
    sim_before = compute_ncc_batch(moving, fixed)
    warped = warp_batch_model_space(moving, theta_pred, mode="bilinear")
    sim_after = compute_ncc_batch(warped, fixed)
    reject = sim_after < sim_before - tolerance
    out = theta_pred.copy()
    out[reject] = IDENTITY_THETA
    return out


# --------------------------------------------------------------------------
# Inference helpers
# --------------------------------------------------------------------------
def compute_reference(data: np.ndarray, reference: str) -> np.ndarray:
    if reference == "average":
        return data.mean(axis=-1)
    elif reference == "first":
        return data[..., 0].copy()
    else:
        raise ValueError(f"Unknown reference strategy: {reference!r}")


def refine_stage2(model2, moving_ms, theta_cum, fixed1, reference, reference2, max_iter, tol, non_reg_tol,
                  stage2_reference="average"):
    """
    Stage 2 refinement with early stopping (tolerance or divergence) and per-slice
    non-regression filter. Batched over all (volume, slice) pairs, except for
    reference2="running", which is sequential in time by definition (batched over slices).
    """
    T, Zc, H, W = moving_ms.shape
    theta_cum = theta_cum.copy()
    flat = lambda a: a.reshape((T * Zc,) + a.shape[2:])

    def warp_all(th):
        return warp_batch_model_space(flat(moving_ms), flat(th), mode="bilinear").reshape(T, Zc, H, W)

    current = warp_all(theta_cum)
    prev_mean_dev = float('inf')
    best_theta_cum = theta_cum.copy()

    for it in range(max_iter):
        residuals = np.broadcast_to(IDENTITY_THETA, theta_cum.shape).copy()

        if reference2 == "running":
            running_sum = current[0].copy()
            for t in range(1, T):
                ref = running_sum / t
                res = predict_theta(model2, current[t], ref)
                res = non_regression_filter_batch(current[t], ref, res, tolerance=non_reg_tol)
                residuals[t] = res
                theta_cum[t] = compose_thetas(theta_cum[t], res)
                current[t] = warp_batch_model_space(moving_ms[t], theta_cum[t], mode="bilinear")
                running_sum += current[t]
        else:
            if reference2 == "same":
                ref = fixed1
            else:
                # Original code: `reference = "first"` here, whatever was asked (kept by default).
                ref_strategy = "average"
                print(f"  Stage 2 iteration {it + 1}/{max_iter}: recomputing reference using '{ref_strategy}' strategy.")
                ref = compute_reference(np.moveaxis(current, 0, -1), ref_strategy)
            ref_all = np.broadcast_to(ref[None], current.shape)
            res = predict_theta(model2, flat(current), flat(ref_all))
            res = non_regression_filter_batch(flat(current), flat(ref_all), res, tolerance=non_reg_tol)
            residuals = res.reshape(T, Zc, 2, 3)

            theta_cum = compose_thetas(theta_cum, residuals)
            current = warp_all(theta_cum)

        mean_dev = float(np.abs(residuals - IDENTITY_THETA).mean())
        print(f"  stage 2 iteration {it + 1}/{max_iter}: mean |residual - identity| = {mean_dev:.5f}")

        if mean_dev < tol:
            print(f"  Converged (< tol={tol}), stopping early.")
            best_theta_cum = theta_cum.copy()
            break

        if it > 0 and mean_dev > prev_mean_dev:
            print("  Residuals increasing (divergence), stopping early.")
            break

        prev_mean_dev = mean_dev
        best_theta_cum = theta_cum.copy()

    return best_theta_cum


# --------------------------------------------------------------------------
# Inference Pipeline
# --------------------------------------------------------------------------
def run_inference(raw_input_path: str, checkpoint_path: str, checkpoint2_path: str, output_path: str,
                  reference: str = "average", reference2: str = "recompute",
                  n_iter2: int = 3, tol2: float = 1e-3, non_reg_tol: float = 0.0,
                  interpolation: str = "bilinear",
                  cropped_path: str = None, keep_cropped: bool = False,
                  stage2_reference: str = "mean"):
    """Returns a dict with the seconds spent in each step (crop, load, preprocess,
    stage1, stage2, final_warp, save)."""
    times = {}
    tic = time.perf_counter()

    def lap(name):
        nonlocal tic
        now = time.perf_counter()
        times[name] = now - tic
        tic = now

    cleanup_cropped = False
    if cropped_path is None:
        fd, cropped_path = tempfile.mkstemp(suffix="_sc_crop.nii.gz")
        os.close(fd)
        os.remove(cropped_path)  # sc_crop writes it
        cleanup_cropped = not keep_cropped
    run_sc_crop(raw_input_path, cropped_path)
    lap("crop")

    raw_img = nib.load(raw_input_path)
    raw_data = raw_img.get_fdata(dtype=np.float32)
    cropped_img = nib.load(cropped_path)
    cropped_data = cropped_img.get_fdata(dtype=np.float32)

    X_raw, Y_raw, Z_raw, T_raw = raw_data.shape
    Xc, Yc, Zc, Tc = cropped_data.shape
    if Tc != T_raw:
        raise ValueError(f"Timepoint count mismatch between raw ({T_raw}) and cropped ({Tc})")

    axis0_offset, axis1_offset, axis2_offset = get_crop_offset_voxels(raw_img, cropped_img)
    crop_center_in_raw_px = (axis1_offset + Yc / 2.0, axis0_offset + Xc / 2.0)
    oz = axis2_offset
    lap("load_data")

    model1 = get_model(checkpoint_path)
    model2 = get_model(checkpoint2_path)
    lap("load_models")

    th, tw = TARGET_SHAPE
    # (X, Y, Z, T) -> (T, Z, X, Y), normalised per slice, padded/cropped to TARGET_SHAPE
    moving_ms = to_model_space_stack(normalize_stack(np.transpose(cropped_data, (3, 2, 0, 1))))
    reference_cropped = compute_reference(cropped_data, reference)
    fixed1 = to_model_space_stack(normalize_stack(np.transpose(reference_cropped, (2, 0, 1))))
    lap("preprocess")

    # --- STAGE 1 + non-regression check (all pairs in one batch) ---
    print("Running Stage 1 with non-regression filter...")
    mov = moving_ms.reshape(T_raw * Zc, th, tw)
    fix = np.broadcast_to(fixed1[None], moving_ms.shape).reshape(T_raw * Zc, th, tw)
    pred = predict_theta(model1, mov, fix)
    theta1_all = non_regression_filter_batch(mov, fix, pred, tolerance=non_reg_tol).reshape(T_raw, Zc, 2, 3)
    lap("stage1")

    # --- STAGE 2 + dynamic iterations & non-regression check ---
    print("Running Stage 2 with dynamic convergence & non-regression filter...")
    theta_total_all = refine_stage2(model2, moving_ms, theta1_all, fixed1,
                                    reference, reference2, n_iter2, tol2, non_reg_tol,
                                    stage2_reference=stage2_reference)
    lap("stage2")

    # --- Final warp in raw space (all valid (t, z) pairs batched) ---
    corrected = raw_data.copy()
    pairs = [(t, z) for t in range(T_raw) for z in range(Zc) if 0 <= z + oz < Z_raw]
    if pairs:
        thetas_raw = np.stack([
            np.asarray(transfer_theta_to_raw(theta_total_all[t, z], TARGET_SHAPE,
                                             crop_center_in_raw_px, (X_raw, Y_raw)), dtype=np.float32)
            for t, z in pairs])
        tt = np.array([p[0] for p in pairs])
        zz = np.array([p[1] for p in pairs]) + oz
        slices = np.ascontiguousarray(np.transpose(raw_data[:, :, zz, tt], (2, 0, 1)))   # (N, X, Y)
        warped = warp_batch_model_space(slices, thetas_raw, mode=interpolation)
        corrected[:, :, zz, tt] = np.transpose(warped, (1, 2, 0))
    lap("final_warp")

    out_img = nib.Nifti1Image(corrected.astype(np.float32), affine=raw_img.affine, header=raw_img.header)
    nib.save(out_img, output_path)
    print(f"Saved motion-corrected volume to {output_path}")

    if cleanup_cropped and os.path.exists(cropped_path):
        os.remove(cropped_path)
    lap("save")
    return times


def main():
    parser = argparse.ArgumentParser(description="Two-stage motion correction with non-regression and dynamic early stopping.")
    parser.add_argument("--input", required=True, help="Path to RAW 4D NIfTI")
    parser.add_argument("--checkpoint1", required=True, help="Path to model 1 checkpoint (.pt)")
    parser.add_argument("--checkpoint2", required=True, help="Path to model 2 checkpoint (.pt)")
    parser.add_argument("--output", required=True, help="Path to output corrected NIfTI")
    parser.add_argument("--reference", choices=["average", "first"], default="average")
    parser.add_argument("--reference2", choices=["recompute", "same", "running"], default="recompute")
    parser.add_argument("--stage2-reference", choices=["first", "average"], default="first",
                        help="Reference recomputed at each stage-2 iteration with --reference2 recompute. "
                             "'first' reproduces the original code; 'average' uses the temporal mean.")
    parser.add_argument("--n-iter2", type=int, default=3, help="Maximum cap for stage-2 iterations (default: 3).")
    parser.add_argument("--tol2", type=float, default=1e-3, help="Convergence threshold. Default 1e-3.")
    parser.add_argument("--non-reg-tol", type=float, default=0.0, help="Tolerance margin for non-regression check (default: 0.0).")
    parser.add_argument("--interpolation", choices=["bilinear", "nearest", "bicubic"], default="nearest")
    parser.add_argument("--cropped-path", default=None)
    parser.add_argument("--keep-cropped", action="store_true")
    args = parser.parse_args()

    times = run_inference(args.input, args.checkpoint1, args.checkpoint2, args.output,
                          reference=args.reference, reference2=args.reference2,
                          n_iter2=args.n_iter2, tol2=args.tol2, non_reg_tol=args.non_reg_tol,
                          interpolation=args.interpolation,
                          cropped_path=args.cropped_path, keep_cropped=args.keep_cropped,
                          stage2_reference=args.stage2_reference)
    print("Time per step (s): " + ", ".join(f"{k} {v:.2f}" for k, v in times.items()))


if __name__ == "__main__":
    main()