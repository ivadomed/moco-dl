# -*- coding: utf-8 -*-
"""
Metric evaluation script for motion-corrected results.
This script evaluates motion correction quality for dMRI and fMRI data
by computing quantitative metrics before and after registration.

Features:
------------
- Automatically locates the moving, fixed, warped, mask, and segmentation files
  using patterns defined in config.yaml.
- Computes:
    • RMSE (Root Mean Squared Error) per timepoint inside the mask
    • SSIM (Structural Similarity Index) inside the mask
    • fMRI-only metrics: temporal SNR (tSNR) and DVARS
- Aggregates all subject/session metrics into a summary CSV

Usage:
------
In terminal/command line:

    python metrics.py /path/to/dataset mode

Arguments:
    /path/to/dataset : path to dmri_dataset or fmri_dataset folder
    mode          : 'dmri' or 'fmri' depending on dataset type
"""

import os
import sys
import csv
import glob
import numpy as np
import nibabel as nib
import torch
torch.set_float32_matmul_precision("medium")

from moco_main import normalize_volume
from skimage.metrics import structural_similarity as ssim
from config_loader import config

# -----------------------------
# Helper functions
# -----------------------------
def load_nifti(path):
    """Load a NIfTI image as numpy array."""
    img = nib.load(path)
    data = img.get_fdata().astype(np.float32)
    return data, img.affine, img.header

def to_tensor(arr, add_dims=True):
    """Convert numpy array to torch tensor with batch/channel dimensions."""
    t = torch.tensor(arr, dtype=torch.float32)
    if add_dims:
        t = t.unsqueeze(0).unsqueeze(0)  # (1,1,H,W,D)
    return t

def compute_rmse(pred, target, mask):
    """Compute Root Mean Squared Error (RMSE) within a binary mask."""
    diff = (pred - target) ** 2
    masked = diff * mask
    denom = mask.sum()
    return torch.sqrt(masked.sum() / (denom + 1e-6)).item()

def compute_ssim_3d(pred, target, mask):
    """
    Compute 3D SSIM between pred and target within a mask.
    SSIM is computed slice-wise along z and averaged for stability.
    """
    pred = pred * (mask > 0)
    target = target * (mask > 0)
    nz = pred.shape[2]
    ssim_vals = []
    for z in range(nz):
        ssim_val = ssim(pred[:, :, z], target[:, :, z],
                        data_range=target.max() - target.min(),
                        gaussian_weights=True, use_sample_covariance=False)
        ssim_vals.append(ssim_val)
    return float(np.mean(ssim_vals))

def compute_tsnr(data, mask):
    """Compute temporal Signal-to-Noise Ratio (tSNR)."""
    mean_t = np.mean(data, axis=-1)
    std_t = np.std(data, axis=-1)
    tsnr_map = np.divide(mean_t, std_t, out=np.zeros_like(mean_t), where=std_t > 0)
    mean_tsnr = np.mean(tsnr_map[mask > 0])
    return tsnr_map, mean_tsnr

def compute_dvars(data, mask):
    """Compute DVARS time series and mean value."""
    masked = data[mask > 0]
    norm_data = masked / np.mean(masked, axis=0, keepdims=True)
    diff = np.diff(norm_data, axis=1)
    dvars = np.sqrt(np.mean(diff ** 2, axis=0))
    mean_dvars = np.mean(dvars)
    return dvars, mean_dvars

# -----------------------------
# Main evaluation
# -----------------------------
def main(data_dir, mode):
    """Main evaluation routine for computing RMSE, SSIM, tSNR, and DVARS."""
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    if mode not in ["dmri", "fmri"]:
        raise ValueError("Mode must be either 'dmri' or 'fmri'")

    patterns = config[mode]
    subjects = sorted(glob.glob(os.path.join(data_dir, "sub-*")))
    print(f"Found {len(subjects)} subjects in {mode} dataset")

    results = []  # store per-subject metrics

    for sub in subjects:
        sessions = sorted(glob.glob(os.path.join(sub, "ses-*")))
        targets = sessions if sessions else [sub]

        for target in targets:
            subdir = os.path.join(target, patterns["subdir"])
            prefix = os.path.basename(target.rstrip("/"))
            print(f"\nProcessing {target}")

            raw_files = glob.glob(os.path.join(subdir, patterns["raw"]))
            moving_files = glob.glob(os.path.join(subdir, patterns["moving"]))
            warped_files = glob.glob(os.path.join(subdir, patterns["moco"]))
            mask_files = glob.glob(os.path.join(subdir, patterns["mask"]))

            if not (moving_files and raw_files and warped_files and mask_files):
                print(f"Missing required files in {target}, skipping.")
                continue

            # -----------------------------
            # Load data
            # -----------------------------
            raw, _, _ = load_nifti(raw_files[0])
            moving, _, _ = load_nifti(moving_files[0])
            warped, aff, hdr = load_nifti(warped_files[0])
            mask, _, _ = load_nifti(mask_files[0])
            mask_t = to_tensor(mask)

            # -----------------------------
            # RMSE / SSIM per timepoint
            # -----------------------------
            T = moving.shape[-1]
            rmse_before, rmse_after = [], []
            ssim_before, ssim_after = [], []

            for t in range(T):
                mov_t = moving[..., t]
                war_t = warped[..., t]
                raw_t = raw[..., t]

                # Normalize inside the mask
                mov_t_n = normalize_volume(to_tensor(mov_t), mask_t).squeeze().numpy()
                war_t_n = normalize_volume(to_tensor(war_t), mask_t).squeeze().numpy()
                raw_t_n = normalize_volume(to_tensor(raw_t), mask_t).squeeze().numpy()

                # ---- RMSE ----
                rmse_before.append(compute_rmse(torch.tensor(mov_t_n), torch.tensor(raw_t_n), mask_t))
                rmse_after.append(compute_rmse(torch.tensor(war_t_n), torch.tensor(raw_t_n), mask_t))

                # ---- SSIM ----
                ssim_before.append(compute_ssim_3d(mov_t_n, raw_t_n, mask))
                ssim_after.append(compute_ssim_3d(war_t_n, raw_t_n, mask))

            mean_rmse_before, mean_rmse_after = np.mean(rmse_before), np.mean(rmse_after)
            mean_ssim_before, mean_ssim_after = np.mean(ssim_before), np.mean(ssim_after)

            print(f"RMSE  before={mean_rmse_before:.4f}, after={mean_rmse_after:.4f}")
            print(f"SSIM  before={mean_ssim_before:.4f}, after={mean_ssim_after:.4f}")

            # -----------------------------
            # fMRI metrics (tSNR, DVARS)
            # -----------------------------
            if mode == "fmri":
                print("Computing tSNR and DVARS ...")
                _, tsnr_raw = compute_tsnr(raw, mask)
                _, tsnr_mov = compute_tsnr(moving, mask)
                _, tsnr_moco  = compute_tsnr(warped, mask)
                _, dvars_raw = compute_dvars(raw, mask)
                _, dvars_mov = compute_dvars(moving, mask)
                _, dvars_moco  = compute_dvars(warped, mask)
                print(f"tSNR  raw={tsnr_raw:.2f}, mov={tsnr_mov:.2f}, moco={tsnr_moco:.2f}")
                print(f"DVARS raw={dvars_raw:.4f}, mov={dvars_mov:.2f}, moco={dvars_moco:.4f}")
            else:
                tsnr_raw = tsnr_mov = tsnr_moco = np.nan
                dvars_raw = dvars_mov = dvars_moco = np.nan

            # -----------------------------
            # Store results
            # -----------------------------
            results.append({
                "SUBJECT": prefix,
                "RMSE_before": mean_rmse_before,
                "RMSE_after": mean_rmse_after,
                "SSIM_before": mean_ssim_before,
                "SSIM_after": mean_ssim_after,
                "tSNR_raw": tsnr_raw,
                "tSNR_mov": tsnr_mov,
                "tSNR_moco": tsnr_moco,
                "DVARS_raw": dvars_raw,
                "DVARS_mov": dvars_mov,
                "DVARS_moco": dvars_moco
            })

    # -----------------------------
    # Save combined summary CSV
    # -----------------------------
    summary_csv = os.path.join(data_dir, f"{mode}_metrics.csv")
    print(f"\nSaving combined metrics summary → {summary_csv}")

    with open(summary_csv, "w", newline="") as f:
        fieldnames = [
            "SUBJECT",
            "RMSE_before", "RMSE_after",
            "SSIM_before", "SSIM_after",
            "tSNR_raw", "tSNR_mov", "tSNR_moco",
            "DVARS_raw", "DVARS_mov", "DVARS_moco"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        for r in results:
            for k, v in r.items():
                if isinstance(v, (float, np.floating)):
                    r[k] = "nan" if np.isnan(v) else f"{v:.4f}"

        writer.writeheader()
        writer.writerows(results)

    print(f"Combined metrics saved → {summary_csv}")

# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python metrics.py <dataset_dir> <mode>")
        sys.exit(1)

    data_dir = sys.argv[1]
    mode = sys.argv[2].lower()
    main(data_dir, mode)