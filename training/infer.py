#!/usr/bin/env python3
"""
moco-dl inference: apply a trained model to test-set subjects.

For each subject in the testing split of dataset.json, this script:

  1. Loads the raw 4D volume, the fixed reference, and the cord mask
     (paths come from dataset.json, written by dataset_preparation.py).
  2. Runs the trained model in forward-only mode at full T.
  3. Warps the volume slice-by-slice using the predicted (Tx, Ty, Theta).
  4. Optionally refines: recompute a sharper fixed from the corrected
     volume, predict again, repeat. ~2-3 iterations is usually plenty.
  5. Saves the corrected volume, the predicted Tx/Ty/Theta fields, and a
     per-subject metrics CSV (tSNR, DVARS, mean displacement, before/after).

Usage
-----
    python infer.py \\
        --checkpoint moco_project/trained_weights/fmri_v1.ckpt \\
        --dataset    data/raw_data/fmri/all_fmri_prepared \\
        --output     moco_project/predictions/fmri_v1 \\
        [--cuda-device 0] [--n-iterations 2] [--subjects sub-XX ...]

Output layout
-------------
    <output>/
        sub-XX/func/
            moco_<original>.nii.gz       (corrected 4D volume — each timepoint warped)
            Tx_<original>.nii.gz         (predicted x-translation field, (D, T))
            Ty_<original>.nii.gz         (predicted y-translation field, (D, T))
            Theta_<original>.nii.gz      (predicted rotation field, degrees, (D, T))
            predictions_<original>.npz   (raw (D, T) arrays for plotting)
            rmse_to_mean_<original>.npz  (per-timepoint RMSE-to-mean: before, after)
            traces_<original>.png        (Tx/Ty/Theta vs t)
            slices_<original>.png        (moving|corrected|diff|fixed at max-mask slice)
            rmse_<original>.png          (RMSE-to-mean vs t, before/after — the moco quality plot)
            metrics.json                 (RMSE-to-mean, tSNR, DVARS, motion before/after)
        summary.csv                      (one row per subject)

The primary quality metric here is **rmse_to_mean**: per-timepoint RMSE between
each frame and the temporal mean (inside the cord mask). After moco, this should
be lower than before at most timepoints. The slice diff column and the RMSE plot
are the fastest visual ways to see whether the model is doing anything.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import warnings
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from train import MocoTrainer

warnings.filterwarnings("ignore", category=FutureWarning)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_nifti(path: Path) -> tuple[np.ndarray, np.ndarray, nib.Nifti1Header]:
    """Return (data_float32, affine, header) for a NIfTI file."""
    img = nib.load(str(path))
    return img.get_fdata().astype(np.float32), img.affine, img.header


def save_like(data: np.ndarray, ref_affine: np.ndarray,
              ref_header: nib.Nifti1Header, out_path: Path) -> None:
    """Save `data` as a NIfTI using the affine/header of a reference image."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = nib.Nifti1Image(data.astype(np.float32), ref_affine, ref_header)
    nib.save(img, str(out_path))


def _save_traces_png(Tx: np.ndarray, Ty: np.ndarray, Theta: np.ndarray,
                     out_path: Path, title: str) -> None:
    """
    Plot mean ± std (over z) of predicted Tx, Ty, Theta vs timepoint.
    Tx, Ty in voxels; Theta in degrees. (D, T) arrays.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(9, 6), sharex=True)
    for ax, arr, ylabel in [
        (axes[0], Tx,    "Tx (axis-0, vox)"),
        (axes[1], Ty,    "Ty (axis-1, vox)"),
        (axes[2], Theta, "Theta (deg)"),
    ]:
        mu = arr.mean(axis=0)
        sd = arr.std(axis=0)
        xs = np.arange(arr.shape[1])
        ax.plot(xs, mu, color="C0", lw=1.5, label="mean over z")
        ax.fill_between(xs, mu - sd, mu + sd, color="C0", alpha=0.25,
                        label="±1 std (across z)")
        ax.axhline(0, color="k", lw=0.5, alpha=0.4)
        ax.grid(alpha=0.3)
        ax.set_ylabel(ylabel)
        if ax is axes[0]:
            ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("timepoint")
    fig.suptitle(f"Predicted motion — {title}", fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def _save_slices_png(moving: np.ndarray, corrected: np.ndarray,
                     fixed: np.ndarray, mask: np.ndarray,
                     out_path: Path, title: str) -> None:
    """
    4 timepoints × 4 columns at the slice with maximum mask coverage:
        moving | corrected | (corrected - moving) | fixed

    The third column is the visual sanity check: even sub-voxel corrections
    show up clearly as a structured difference image. A no-op model produces
    a near-zero difference column.

    Mask outline overlaid in red on every panel for orientation.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    H, W, D, T = moving.shape
    if mask.any():
        z = int(mask.sum(axis=(0, 1)).argmax())
    else:
        z = D // 2

    t_idx = [int(round(x)) for x in np.linspace(0, T - 1, 4)]
    fixed_per_t = (fixed.ndim == 4)

    fig, axes = plt.subplots(4, 4, figsize=(10, 9))
    # Intensity range from fixed; symmetric range for the diff column
    fixed_view = fixed[:, :, z, 0] if fixed_per_t else fixed[:, :, z]
    vmin, vmax = np.percentile(fixed_view, (1, 99))
    diff_vol = corrected - moving
    diff_at_z = diff_vol[:, :, z, :]
    # Use a robust symmetric range so the colormap centers on zero
    dmax = float(np.percentile(np.abs(diff_at_z), 99)) or 1e-6
    mask_2d = mask[:, :, z] if mask.any() else None

    for row, t in enumerate(t_idx):
        panels = [
            ("moving",    moving[:, :, z, t],    dict(cmap="gray", vmin=vmin, vmax=vmax)),
            ("corrected", corrected[:, :, z, t], dict(cmap="gray", vmin=vmin, vmax=vmax)),
            ("corr - mov", diff_at_z[:, :, t],    dict(cmap="RdBu_r", vmin=-dmax, vmax=dmax)),
            ("fixed",     fixed[:, :, z, t] if fixed_per_t else fixed[:, :, z],
                          dict(cmap="gray", vmin=vmin, vmax=vmax)),
        ]
        for col, (name, img, kw) in enumerate(panels):
            ax = axes[row, col]
            ax.imshow(img.T, origin="lower", **kw)
            if mask_2d is not None and mask_2d.any():
                ax.contour(mask_2d.T, levels=[0.5], colors="red",
                           linewidths=0.5, alpha=0.6)
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(name, fontsize=10)
            if col == 0:
                ax.set_ylabel(f"t={t}", fontsize=9)
    fig.suptitle(f"{title}  z={z} (max-mask slice)  diff range ±{dmax:.2g}", fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def _save_rmse_png(before_rmse_t: np.ndarray, after_rmse_t: np.ndarray,
                   out_path: Path, title: str) -> None:
    """
    Per-timepoint RMSE between each frame and the temporal mean, before vs after.
    The direct moco quality plot: after should be lower than before at every t,
    on average if the model is helping.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 3.5))
    xs = np.arange(len(before_rmse_t))
    ax.plot(xs, before_rmse_t, color="C3", lw=1.2, label="before moco")
    ax.plot(xs, after_rmse_t, color="C2", lw=1.2, label="after moco")
    ax.fill_between(xs, after_rmse_t, before_rmse_t,
                    where=after_rmse_t < before_rmse_t,
                    color="C2", alpha=0.15, label="improvement")
    ax.fill_between(xs, after_rmse_t, before_rmse_t,
                    where=after_rmse_t >= before_rmse_t,
                    color="C3", alpha=0.15, label="regression")
    ax.set_xlabel("timepoint")
    ax.set_ylabel("RMSE to temporal mean (inside mask)")
    ax.set_title(f"{title}  —  alignment quality per timepoint")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def tsnr(vol4d: np.ndarray, mask3d: np.ndarray) -> float:
    """Temporal SNR averaged inside mask. Higher = less temporal noise."""
    m = mask3d > 0
    if not m.any():
        return float("nan")
    mu = vol4d.mean(axis=-1)
    sd = vol4d.std(axis=-1)
    tsnr_map = np.where(sd > 1e-6, mu / (sd + 1e-6), 0.0)
    return float(tsnr_map[m].mean())


def dvars(vol4d: np.ndarray, mask3d: np.ndarray) -> float:
    """
    DVARS = root-mean-square temporal derivative of intensities inside mask.
    Standard fMRI motion metric. Lower = less frame-to-frame variation.
    """
    m = mask3d > 0
    if not m.any() or vol4d.shape[-1] < 2:
        return float("nan")
    diff = np.diff(vol4d, axis=-1)                  # (H, W, D, T-1)
    diff_masked = diff[m]                            # (Nvox, T-1)
    return float(np.sqrt((diff_masked ** 2).mean()))


def mean_displacement(Tx: np.ndarray, Ty: np.ndarray, Theta: np.ndarray | None = None
                      ) -> float:
    """
    Mean per-(z, t) magnitude of the predicted translation, in voxels.
    A summary of "how much motion the model believes it corrected."
    """
    if Tx is None or Ty is None:
        return float("nan")
    mag = np.sqrt(Tx ** 2 + Ty ** 2)
    return float(mag.mean())


def rmse_to_mean(vol4d: np.ndarray, mask3d: np.ndarray) -> np.ndarray:
    """
    Per-timepoint RMSE between each frame and the temporal mean of vol4d,
    restricted to mask3d. Shape (T,). Lower = better alignment to the mean.

    This is the direct moco quality metric: a perfectly aligned series has
    every frame == mean (modulo physiological / thermal noise), so RMSE→0.
    Motion makes each frame deviate from the mean, so RMSE>0.
    """
    m = mask3d > 0
    if not m.any() or vol4d.shape[-1] < 2:
        return np.full(vol4d.shape[-1], np.nan, dtype=np.float32)
    mean3d = vol4d.mean(axis=-1)            # (H, W, D)
    diffs = vol4d - mean3d[..., None]       # (H, W, D, T)
    diffs_masked = diffs[m]                  # (Nvox, T)
    return np.sqrt((diffs_masked ** 2).mean(axis=0)).astype(np.float32)


# ---------------------------------------------------------------------------
# Inference for one subject
# ---------------------------------------------------------------------------

def _build_fixed_from(volume4d: np.ndarray, original_fixed: np.ndarray
                       ) -> np.ndarray:
    """
    Compute a refined fixed reference from a (possibly corrected) 4D volume.

    For fMRI (original_fixed is 3D): use the temporal mean of the corrected
    volume. After moco, this mean is sharper than the original.

    For dMRI (original_fixed is 4D, per-timepoint): keep the original
    fixed structure (duplicated mean-b0 / mean-dwi). Recomputing it
    requires the bval file which we don't have at this stage — sticking
    with the original is safe and matches what `preprocessing.py` built.
    """
    if original_fixed.ndim == 3:
        return volume4d.mean(axis=-1).astype(np.float32)
    return original_fixed


@torch.no_grad()
def predict_translations(model: MocoTrainer, moving: np.ndarray,
                          fixed: np.ndarray, device: torch.device
                          ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Forward pass — returns (Tx, Ty, Theta) each of shape (D, T), all float32.
    No autograd, so memory is dominated by inputs and one set of feature maps.
    """
    H, W, D, T = moving.shape
    mov_t = torch.from_numpy(moving).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,H,W,D,T)
    if fixed.ndim == 3:
        fix_t = torch.from_numpy(fixed).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,H,W,D)
    else:
        fix_t = torch.from_numpy(fixed).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,H,W,D,T)

    Tx, Ty, Theta = model.model(mov_t, fix_t)         # each (1, D, T)
    return (Tx.squeeze(0).cpu().numpy(),
            Ty.squeeze(0).cpu().numpy(),
            Theta.squeeze(0).cpu().numpy())


@torch.no_grad()
def warp_volume(model: MocoTrainer, moving: np.ndarray,
                Tx: np.ndarray, Ty: np.ndarray, Theta: np.ndarray,
                device: torch.device) -> np.ndarray:
    """Apply the predicted (Tx, Ty, Theta) to the moving volume timepoint-by-timepoint."""
    H, W, D, T = moving.shape
    mov_t = torch.from_numpy(moving).unsqueeze(0).unsqueeze(0).to(device)
    Tx_t = torch.from_numpy(Tx).unsqueeze(0).to(device)        # (1, D, T)
    Ty_t = torch.from_numpy(Ty).unsqueeze(0).to(device)
    Th_t = torch.from_numpy(Theta).unsqueeze(0).to(device)

    out = torch.empty_like(mov_t)
    for t in range(T):
        out[..., t] = model.warp(mov_t[..., t],
                                 Tx_t[..., t], Ty_t[..., t], Th_t[..., t])
    return out.squeeze(0).squeeze(0).cpu().numpy()


def infer_subject(model: MocoTrainer, raw_path: Path, fixed_path: Path,
                  mask_path: Path, out_dir: Path,
                  n_iterations: int, device: torch.device,
                  ) -> dict:
    """Process one subject end-to-end. Returns a metrics dict."""
    print(f"  Loading {raw_path.name}")
    moving, affine, header = load_nifti(raw_path)
    fixed, _, _ = load_nifti(fixed_path)
    mask, _, _ = load_nifti(mask_path)

    # The original raw volume — what we measure "before" against.
    # The "alignment quality" metric is the per-timepoint RMSE between each
    # frame and the temporal mean inside the mask: this directly measures
    # how aligned the series is to its own mean. After moco, this should drop.
    before_tsnr = tsnr(moving, mask)
    before_dvars = dvars(moving, mask)
    before_rmse_t = rmse_to_mean(moving, mask)         # (T,)
    print(f"  Before: tSNR={before_tsnr:.2f}  DVARS={before_dvars:.4f}  "
          f"RMSE-to-mean (mean over t)={before_rmse_t.mean():.4f}  "
          f"shape={moving.shape}  T={moving.shape[-1]}")

    current = moving.copy()
    current_fixed = fixed.copy()
    last_Tx = last_Ty = last_Theta = None

    for it in range(n_iterations):
        t0 = time.time()
        Tx, Ty, Theta = predict_translations(model, current, current_fixed, device)
        warped = warp_volume(model, current, Tx, Ty, Theta, device)
        elapsed = time.time() - t0
        disp = mean_displacement(Tx, Ty)
        print(f"  Iter {it+1}/{n_iterations}  "
              f"mean |T|={disp:.3f} vox  max |T|={float(np.sqrt(Tx**2+Ty**2).max()):.3f} vox  "
              f"({elapsed:.1f}s)")
        current = warped
        last_Tx, last_Ty, last_Theta = Tx, Ty, Theta
        # Refresh the fixed for the next iteration (helps for fMRI)
        if it < n_iterations - 1:
            current_fixed = _build_fixed_from(current, fixed)

    # After-moco metrics
    after_tsnr = tsnr(current, mask)
    after_dvars = dvars(current, mask)
    after_rmse_t = rmse_to_mean(current, mask)
    print(f"  After:  tSNR={after_tsnr:.2f}  DVARS={after_dvars:.4f}  "
          f"RMSE-to-mean (mean over t)={after_rmse_t.mean():.4f}")
    print(f"  Delta:  tSNR={after_tsnr - before_tsnr:+.2f}  "
          f"DVARS={after_dvars - before_dvars:+.4f}  "
          f"RMSE={after_rmse_t.mean() - before_rmse_t.mean():+.4f}")

    # ----- Save outputs -----
    stem = raw_path.name.replace(".nii.gz", "").replace(".nii", "")
    save_like(current, affine, header,
              out_dir / f"moco_{raw_path.name}")

    # Save the predicted fields. Shape (D, T) — z along axis 0, t along axis 1.
    # Loadable in any NIfTI viewer / numpy script for plotting.
    def _save_field(arr: np.ndarray, name: str) -> None:
        save_like(arr.astype(np.float32),
                  ref_affine=np.eye(4),
                  ref_header=nib.Nifti1Header(),
                  out_path=out_dir / f"{name}_{stem}.nii.gz")
    _save_field(last_Tx, "Tx")
    _save_field(last_Ty, "Ty")
    _save_field(last_Theta, "Theta")
    # Also save the raw arrays as one .npz for convenient Python loading
    np.savez(out_dir / f"predictions_{stem}.npz",
             Tx=last_Tx, Ty=last_Ty, Theta=last_Theta)

    # ----- Diagnostic PNGs (lazy imports so matplotlib isn't required for import) -----
    _save_traces_png(last_Tx, last_Ty, last_Theta,
                     out_dir / f"traces_{stem}.png", stem)
    _save_slices_png(moving, current, fixed, mask,
                     out_dir / f"slices_{stem}.png", stem)
    _save_rmse_png(before_rmse_t, after_rmse_t,
                   out_dir / f"rmse_{stem}.png", stem)
    # Also save the per-t RMSE arrays for downstream analysis
    np.savez(out_dir / f"rmse_to_mean_{stem}.npz",
             before=before_rmse_t, after=after_rmse_t)

    metrics = {
        "subject":       stem,
        "shape":         list(moving.shape),
        "n_iterations":  n_iterations,
        "tsnr_before":   before_tsnr,
        "tsnr_after":    after_tsnr,
        "tsnr_delta":    after_tsnr - before_tsnr,
        "dvars_before":  before_dvars,
        "dvars_after":   after_dvars,
        "dvars_delta":   after_dvars - before_dvars,
        "rmse_to_mean_before":  float(before_rmse_t.mean()),
        "rmse_to_mean_after":   float(after_rmse_t.mean()),
        "rmse_to_mean_delta":   float(after_rmse_t.mean() - before_rmse_t.mean()),
        "rmse_to_mean_relative": float(
            (after_rmse_t.mean() - before_rmse_t.mean()) / max(before_rmse_t.mean(), 1e-8)
        ),
        "mean_displacement_vox": float(np.sqrt(last_Tx ** 2 + last_Ty ** 2).mean()),
        "max_displacement_vox":  float(np.sqrt(last_Tx ** 2 + last_Ty ** 2).max()),
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="Path to a .ckpt produced by train.py")
    p.add_argument("--dataset", type=Path, required=True,
                   help="Prepared dataset directory containing dataset.json")
    p.add_argument("--output", type=Path, required=True,
                   help="Where to write moco_*.nii.gz and metrics")
    p.add_argument("--subjects", nargs="+", default=None,
                   help="Only process subjects whose path contains one of these strings "
                        "(e.g. sub-04 ds004386_sub-12). Default: all test subjects.")
    p.add_argument("--n-iterations", type=int, default=2,
                   help="Number of predict-warp-refine iterations. 2-3 helps for fMRI; "
                        "1 is the simplest case.")
    p.add_argument("--cuda-device", type=str, default=None,
                   help="Set CUDA_VISIBLE_DEVICES (e.g. '0' or '1').")
    p.add_argument("--cpu", action="store_true",
                   help="Force CPU. Useful for debugging.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.cuda_device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device

    # Load dataset.json — get the test entries
    json_path = args.dataset / "dataset.json"
    if not json_path.exists():
        sys.exit(f"ERROR: {json_path} not found")
    with open(json_path) as f:
        ds = json.load(f)

    test_entries = ds.get("testing", [])
    if not test_entries:
        sys.exit("ERROR: dataset.json has no 'testing' entries")

    if args.subjects:
        test_entries = [e for e in test_entries
                        if any(s in str(e.get("raw", "")) for s in args.subjects)]
        if not test_entries:
            sys.exit(f"ERROR: no testing entries matched --subjects {args.subjects}")

    # Pick device
    if args.cpu or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Dataset:    {args.dataset}")
    print(f"Output:     {args.output}")
    print(f"Test subjects: {len(test_entries)}")

    # Load model
    print(f"\nLoading checkpoint ...")
    model = MocoTrainer.load_from_checkpoint(str(args.checkpoint), map_location=device)
    model = model.to(device).eval()

    # Process subjects
    args.output.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for i, entry in enumerate(test_entries, 1):
        # The entry has relative paths under the dataset dir
        raw_path = args.dataset / entry["raw"]
        fixed_path = args.dataset / entry["fixed"]
        mask_path = args.dataset / entry["mask"]

        # Derive a per-subject output dir
        rel_subdir = Path(entry["raw"]).parent
        out_dir = args.output / rel_subdir

        print(f"\n[{i}/{len(test_entries)}] {raw_path.name}")
        if not raw_path.exists():
            print(f"  [skip] {raw_path} missing")
            continue
        try:
            metrics = infer_subject(model, raw_path, fixed_path, mask_path,
                                    out_dir, args.n_iterations, device)
            summary_rows.append(metrics)
        except Exception as e:
            print(f"  [error] {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()

    # Write summary CSV
    if summary_rows:
        summary_path = args.output / "summary.csv"
        # Pick a stable column order — most useful metrics first.
        # RMSE-to-mean is the primary moco quality metric here: it directly
        # measures how aligned each frame is to the temporal mean.
        cols = ["subject", "n_iterations",
                "rmse_to_mean_before", "rmse_to_mean_after",
                "rmse_to_mean_delta", "rmse_to_mean_relative",
                "tsnr_before", "tsnr_after", "tsnr_delta",
                "dvars_before", "dvars_after", "dvars_delta",
                "mean_displacement_vox", "max_displacement_vox", "shape"]
        with open(summary_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for row in summary_rows:
                w.writerow({k: row[k] for k in cols})
        print(f"\nWrote {summary_path}")

        # Print aggregate
        print("\nAggregate over all subjects:")
        for k in ("rmse_to_mean_delta", "rmse_to_mean_relative",
                  "tsnr_delta", "dvars_delta", "mean_displacement_vox"):
            vals = [r[k] for r in summary_rows]
            print(f"  {k:25s}  mean={np.mean(vals):+.3f}  "
                  f"median={np.median(vals):+.3f}  "
                  f"min/max={min(vals):+.3f}/{max(vals):+.3f}")


if __name__ == "__main__":
    main()