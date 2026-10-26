"""
Dataset inspection script for spinal cord fMRI BOLD NIfTI files.

Discovers ONLY BOLD scans (filename contains '_bold' and ends in .nii/.nii.gz),
robust to BIDS layouts with optional ses- subfolders, and also to the flat
non-BIDS layout used by train_translation_first.py (train/ and validation/
folders containing one 4D NIfTI per subject). Non-BOLD files (T1w, T2w,
anat, fieldmaps, physio, derivatives, etc.) are skipped.

Writes:
  - dataset_summary.csv   : one row per BOLD volume
  - dataset_report.txt    : human-readable summary

Usage
-----
    python inspect_dataset.py
    python inspect_dataset.py --data crop_dataset --output my_report_dir

Requires: nibabel, numpy
"""

import argparse
import csv
import glob
import os
import re
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

DEFAULT_DATA_DIR = "crop_dataset"
DEFAULT_OUT_DIR  = "dataset_inspection"

# BIDS suffix for BOLD: ..._bold.nii[.gz]
BOLD_PATTERN = re.compile(r"_bold\.nii(\.gz)?$", re.IGNORECASE)

# Folders that are never BOLD data — skip them wholesale
EXCLUDE_DIRS = {"anat", "fmap", "dwi", "perf", "derivatives", "sourcedata", "code"}


def is_bold_file(path: str) -> bool:
    """
    Keep only BOLD fMRI files:
      - filename matches *_bold.nii[.gz]  (BIDS suffix)
      - no excluded parent folder (anat/, fmap/, derivatives/, …)
    Also keep bare 4D files in flat (non-BIDS) layouts: if the path has no
    BIDS-style tokens at all (no 'sub-', no 'ses-', no 'func/'), the
    filename is accepted as BOLD since the training script treats every
    file in train/ and validation/ as a BOLD volume.
    """
    name = os.path.basename(path).lower()
    parts = {p.lower() for p in Path(path).parts}

    if parts & EXCLUDE_DIRS:
        return False

    # BIDS-style file: must have the _bold suffix
    bids_like = any(p.startswith(("sub-", "ses-")) or p == "func" for p in parts)
    if bids_like:
        return bool(BOLD_PATTERN.search(name)) and "func" in parts

    # Flat layout (as in crop_dataset/train/*.nii.gz): accept everything
    return True


def discover_bold_volumes(root: str) -> list[dict]:
    """
    Walk `root` recursively and return only BOLD .nii/.nii.gz files.
    Each entry: {path, split, subject_id, session}.
    """
    entries = []
    all_niftis = sorted(
        glob.glob(os.path.join(root, "**", "*.nii.gz"), recursive=True) +
        glob.glob(os.path.join(root, "**", "*.nii"),    recursive=True)
    )
    kept, skipped = [], []
    for path in all_niftis:
        if is_bold_file(path):
            kept.append(path)
        else:
            skipped.append(path)

    for path in kept:
        rel   = os.path.relpath(path, root)
        parts = Path(rel).parts
        split = parts[0] if len(parts) > 1 else "root"

        # Pull BIDS tokens when present
        sub_token = next((p for p in parts if p.startswith("sub-")), None)
        ses_token = next((p for p in parts if p.startswith("ses-")), None)

        if sub_token:
            subject_id = sub_token
        else:
            subject_id = Path(path).stem.replace(".nii", "")

        entries.append({
            "path":       path,
            "split":      split,
            "subject_id": subject_id,
            "session":    ses_token or "",
        })

    # Deduplicate paths
    seen, unique = set(), []
    for e in entries:
        if e["path"] not in seen:
            seen.add(e["path"])
            unique.append(e)

    return unique, skipped


def inspect_volume(entry: dict) -> dict:
    path = entry["path"]
    img  = nib.load(path)
    hdr  = img.header

    shape  = img.shape
    ndim   = len(shape)
    pixdim = hdr.get_zooms()

    n_x = shape[0]
    n_y = shape[1]
    n_z = shape[2] if ndim >= 3 else 1
    n_t = shape[3] if ndim == 4 else 1

    vox_x = float(pixdim[0]) if len(pixdim) > 0 else float("nan")
    vox_y = float(pixdim[1]) if len(pixdim) > 1 else float("nan")
    vox_z = float(pixdim[2]) if len(pixdim) > 2 else float("nan")
    tr_s  = float(pixdim[3]) if ndim == 4 and len(pixdim) > 3 else float("nan")

    file_mb = os.path.getsize(path) / 1_048_576
    data    = img.get_fdata(dtype=np.float32)

    return {
        "split":          entry["split"],
        "subject_id":     entry["subject_id"],
        "session":        entry["session"],
        "path":           path,
        "n_x":            n_x,
        "n_y":            n_y,
        "n_z":            n_z,
        "n_timepoints":   n_t,
        "vox_x_mm":       round(vox_x, 4),
        "vox_y_mm":       round(vox_y, 4),
        "vox_z_mm":       round(vox_z, 4),
        "tr_s":           round(tr_s, 4) if not np.isnan(tr_s) else "",
        "dtype":          str(data.dtype),
        "file_mb":        round(file_mb, 2),
        "mean_intensity": round(float(data.mean()), 4),
        "min_intensity":  round(float(data.min()),  4),
        "max_intensity":  round(float(data.max()),  4),
    }


def write_csv(rows: list[dict], path: str) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_report(rows: list[dict], data_root: str, n_skipped: int) -> str:
    lines = []
    banner = "=" * 72
    lines += [banner, "  SPINAL CORD fMRI BOLD DATASET REPORT", banner, ""]
    lines.append(f"  Data root       : {os.path.abspath(data_root)}")
    lines.append(f"  BOLD volumes    : {len(rows)}")
    lines.append(f"  Non-BOLD skipped: {n_skipped}")
    lines.append("")

    lines.append("── BOLD volumes per split " + "─" * 45)
    for sp in sorted({r["split"] for r in rows}):
        sub = [r for r in rows if r["split"] == sp]
        lines.append(f"  {sp:<24}  {len(sub)} volume(s)")
    lines.append("")

    # Unique subjects
    subs = {r["subject_id"] for r in rows}
    lines.append(f"── Unique subjects: {len(subs)}")
    lines.append("")

    # Shape breakdown
    lines.append("── Spatial shapes (X × Y × Z) " + "─" * 40)
    shapes = {}
    for r in rows:
        key = f"{r['n_x']} × {r['n_y']} × {r['n_z']}"
        shapes[key] = shapes.get(key, 0) + 1
    for k, v in sorted(shapes.items(), key=lambda x: -x[1]):
        lines.append(f"  {k:<32}  {v} volume(s)")
    lines.append("")

    # Timepoints
    lines.append("── Timepoints per volume " + "─" * 46)
    tpts = {}
    for r in rows:
        tpts[r["n_timepoints"]] = tpts.get(r["n_timepoints"], 0) + 1
    for k, v in sorted(tpts.items()):
        lines.append(f"  T = {k:<8}  {v} volume(s)")
    lines.append("")

    # Voxel sizes
    lines.append("── Voxel sizes (mm) " + "─" * 51)
    res_set = {}
    for r in rows:
        key = f"{r['vox_x_mm']:.4f} × {r['vox_y_mm']:.4f} × {r['vox_z_mm']:.4f}"
        res_set[key] = res_set.get(key, 0) + 1
    for k, v in sorted(res_set.items(), key=lambda x: -x[1]):
        lines.append(f"  {k:<42}  {v} volume(s)")
    lines.append("")

    # TR
    lines.append("── TR (s) " + "─" * 61)
    trs = {}
    for r in rows:
        key = str(r["tr_s"]) if r["tr_s"] != "" else "not set"
        trs[key] = trs.get(key, 0) + 1
    for k, v in sorted(trs.items(), key=lambda x: -x[1]):
        lines.append(f"  TR = {k:<12}  {v} volume(s)")
    lines.append("")

    # Data types
    lines.append("── NIfTI dtypes " + "─" * 55)
    dtypes = {}
    for r in rows:
        dtypes[r["dtype"]] = dtypes.get(r["dtype"], 0) + 1
    for k, v in dtypes.items():
        lines.append(f"  {k:<24}  {v} volume(s)")
    lines.append("")

    # File sizes
    mbs = [r["file_mb"] for r in rows]
    lines.append("── File sizes (MB) " + "─" * 52)
    lines.append(f"  min {min(mbs):.1f}   mean {np.mean(mbs):.1f}   "
                 f"max {max(mbs):.1f}   total {sum(mbs):.1f}")
    lines.append("")

    # Per-volume table
    lines.append("── Per-volume detail " + "─" * 50)
    cw = [24, 12, 20, 8, 6, 8, 8, 8]
    hdr = (f"  {'Subject':<{cw[0]}}"
           f"{'Split':<{cw[1]}}"
           f"{'Shape (X×Y×Z×T)':<{cw[2]}}"
           f"{'VoxXY':>{cw[3]}}"
           f"{'TR':>{cw[4]}}"
           f"{'Mean':>{cw[5]}}"
           f"{'Min':>{cw[6]}}"
           f"{'Max':>{cw[7]}}")
    lines.append(hdr)
    lines.append("  " + "─" * (sum(cw) + 2))
    for r in rows:
        shape_str = f"{r['n_x']}×{r['n_y']}×{r['n_z']}×{r['n_timepoints']}"
        tr_str    = str(r["tr_s"]) if r["tr_s"] != "" else "—"
        lines.append(
            f"  {r['subject_id']:<{cw[0]}}"
            f"{r['split']:<{cw[1]}}"
            f"{shape_str:<{cw[2]}}"
            f"{r['vox_x_mm']:>{cw[3]}.3f}"
            f"{tr_str:>{cw[4]}}"
            f"{r['mean_intensity']:>{cw[5]}.2f}"
            f"{r['min_intensity']:>{cw[6]}.2f}"
            f"{r['max_intensity']:>{cw[7]}.2f}"
        )
    lines.append("")
    lines.append(banner)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Inspect fMRI BOLD NIfTI dataset. Non-BOLD files are skipped."
    )
    parser.add_argument("--data", "-d", default=DEFAULT_DATA_DIR,
                        help=f"Root data folder (default: {DEFAULT_DATA_DIR})")
    parser.add_argument("--output", "-o", default=DEFAULT_OUT_DIR,
                        help=f"Output folder (default: {DEFAULT_OUT_DIR})")
    args = parser.parse_args()

    if not os.path.isdir(args.data):
        sys.exit(f"ERROR: data directory not found: {args.data}")

    os.makedirs(args.output, exist_ok=True)

    entries, skipped = discover_bold_volumes(args.data)
    if not entries:
        sys.exit(f"ERROR: no BOLD .nii/.nii.gz files found under {args.data}")

    print(f"Found {len(entries)} BOLD file(s)  "
          f"({len(skipped)} non-BOLD file(s) skipped)\n")

    rows = []
    for i, entry in enumerate(entries, 1):
        print(f"  [{i:>3}/{len(entries)}] {entry['subject_id']} ({entry['split']})", end="  ", flush=True)
        row = inspect_volume(entry)
        rows.append(row)
        print(f"{row['n_x']}×{row['n_y']}×{row['n_z']}×{row['n_timepoints']}  "
              f"vox={row['vox_x_mm']}×{row['vox_y_mm']}×{row['vox_z_mm']} mm  "
              f"TR={row['tr_s']}s")

    csv_path = os.path.join(args.output, "dataset_summary.csv")
    write_csv(rows, csv_path)
    print(f"\nSaved: {csv_path}")

    report = build_report(rows, args.data, len(skipped))
    report_path = os.path.join(args.output, "dataset_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Saved: {report_path}\n")
    print(report)


if __name__ == "__main__":
    main()