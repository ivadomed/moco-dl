#!/usr/bin/env python3
"""Split diffusion MRI (dMRI) images into train/validation by subject and crop them with sc_crop.

Usage:
    python crop_split_dwi.py --data-root DATA_DIR --output OUTPUT_DIR [--files files.txt]

  --data-root   folder given to download_data.sh (contains ds004386/, ds004616/, ...)
  --files       optional list of images, one per line as <dataset>/<path>. If not given,
                every file ending in dwi.nii.gz under DATA_DIR/<dataset>/ is used.
  --output      where train/ and validation/ are written
  --val-size    fraction of SUBJECTS in validation (default: 0.2)
  --seed        random seed (default: 42)
  --crop-cmd    crop command, called as: <crop-cmd> -i INPUT -o OUTPUT (default: sc_crop)
  --include-derivatives  also search inside derivatives/ folders (skipped by default)

sourcedata/ and prepared/ folders are always skipped, since they hold copies of the raw images.

All images of a subject go to the same split, so no subject is in both train and validation.
The split is done within each dataset, so every dataset contributes ~val-size of its subjects.
Output files are named <dataset>_<original BIDS name>_crop.nii.gz. The matching .bval, .bvec
and .json sidecars are copied next to the cropped image with the same name (cropping is spatial
only, so the gradient table is unchanged). The split is saved to OUTPUT_DIR/split.csv so it can
be reused without re-running the split.
"""
import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

SUFFIX = "dwi.nii.gz"
SIDECARS = (".bval", ".bvec", ".json")
SKIP_DIRS = {"sourcedata", "prepared"}  # folders never searched


def make_row(line):
    dataset, rel = line.split("/", 1)
    sub = re.search(r"(?:^|/)sub-([^/_]+)", rel)
    if not sub:
        sys.exit(f"No subject found in: {line}")
    return {"dataset": dataset, "subject": sub.group(1), "file": line}


def read_file_list(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if not line.endswith(SUFFIX):
            continue  # keep only diffusion images
        rows.append(make_row(line))
    return pd.DataFrame(rows)


def find_dwi(data_root, include_derivatives=False):
    """Every *dwi.nii.gz under data_root, as <dataset>/<path> strings."""
    rows = []
    for p in data_root.rglob(f"*{SUFFIX}"):  # also lists git-annex symlinks not yet downloaded
        rel = p.relative_to(data_root)
        parts = rel.parts
        if len(parts) < 2 or any(x.startswith(".") for x in parts):
            continue  # file directly in data_root, or inside .git/ etc.
        if SKIP_DIRS.intersection(parts):
            continue  # copies of the raw data, would duplicate subjects
        if not include_derivatives and "derivatives" in parts:
            continue
        if not re.search(r"(?:^|/)sub-[^/_]+", "/".join(parts[1:])):
            continue  # e.g. a template image, not tied to a subject
        rows.append(make_row(rel.as_posix()))
    return pd.DataFrame(rows, columns=["dataset", "subject", "file"])


def sidecar(path, ext):
    """sub-01_dwi.nii.gz -> sub-01_dwi.bval (same folder)."""
    return path.with_name(path.name[:-len(".nii.gz")] + ext)


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--files", default=None, type=Path)
    ap.add_argument("--output", default=here / "cropped_data_dwi", type=Path)
    ap.add_argument("--val-size", default=0.2, type=float)
    ap.add_argument("--seed", default=42, type=int)
    ap.add_argument("--crop-cmd", default="sc_crop")
    ap.add_argument("--include-derivatives", action="store_true")
    args = ap.parse_args()

    if args.files:
        df = read_file_list(args.files)
    else:
        df = find_dwi(args.data_root, args.include_derivatives)
    if df.empty:
        sys.exit(f"No files ending in {SUFFIX} found")
    df = df.sort_values("file").reset_index(drop=True)
    print(f"Found {len(df)} DWI images in {df.dataset.nunique()} datasets")

    # Split by subject, separately in each dataset so each one is represented in validation
    groups = df.dataset + "/" + df.subject
    df["split"] = "train"
    for _, part in df.groupby("dataset"):
        if part.subject.nunique() < 2:
            continue
        splitter = GroupShuffleSplit(n_splits=1, test_size=args.val_size, random_state=args.seed)
        _, val_idx = next(splitter.split(part, groups=part.subject))
        df.loc[part.index[val_idx], "split"] = "validation"

    # <dataset>_<original name>_crop.nii.gz keeps names unique across datasets
    def out_name(r):
        stem = Path(r.file).name[:-len(".nii.gz")]
        return f"{r.split}/{r.dataset}_{stem}_crop.nii.gz"
    df["output"] = df.apply(out_name, axis=1)
    dup = df[df.output.duplicated(keep=False)].sort_values("output")
    if not dup.empty:
        print(f"[ERROR] {dup.output.nunique()} output names are used by more than one input:")
        for name, part in dup.groupby("output"):
            print(f"  {name}\n    " + "\n    ".join(part.file))
        sys.exit(1)

    args.output.mkdir(parents=True, exist_ok=True)
    df[["split", "dataset", "subject", "file", "output"]].to_csv(args.output / "split.csv", index=False)
    n_sub = groups.groupby(df.split).nunique()
    print(f"train: {(df.split == 'train').sum()} images / {n_sub.get('train', 0)} subjects | "
          f"validation: {(df.split == 'validation').sum()} images / {n_sub.get('validation', 0)} subjects")

    failed = []
    for r in df.itertuples():
        src = args.data_root / r.file
        dst = args.output / r.output

        if not dst.exists():
            if not src.is_file():  # also false for git-annex files that were not downloaded
                print(f"[WARNING] Not found or not downloaded, skipping: {src}")
                failed.append(r.file)
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            print(f"[{r.split.upper()}] {dst.name}")
            try:
                subprocess.run([args.crop_cmd, "-i", str(src), "-o", str(dst)], check=True)
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                print(f"[ERROR] Failed to crop {src}: {e}")
                failed.append(r.file)
                continue

        # Copy .bval/.bvec/.json next to the cropped image with the same name
        for ext in SIDECARS:
            s, d = sidecar(src, ext), sidecar(dst, ext)
            if d.exists():
                continue
            if s.is_file():
                shutil.copyfile(s, d)
            elif ext != ".json":
                print(f"[WARNING] Missing {ext} for {src} (may be inherited at a higher level)")

    print(f"Done: {len(df) - len(failed)}/{len(df)} images in {args.output}")
    if failed:
        print(f"{len(failed)} failed:\n  " + "\n  ".join(failed))
        sys.exit(1)


if __name__ == "__main__":
    main()