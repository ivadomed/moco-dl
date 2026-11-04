#!/usr/bin/env python3
"""Split the downloaded images into train/validation by subject and crop them with sc_crop.

Usage:
    python crop_split_dataset.py --data-root DATA_DIR --output OUTPUT_DIR [--files files.txt]

  --data-root   folder given to download_data.sh (contains ds004386/, ds004616/, ...)
  --files       list of images, one per line as <dataset>/<path> (default: files.txt next to this script)
  --output      where train/ and validation/ are written
  --val-size    fraction of SUBJECTS in validation (default: 0.2)
  --seed        random seed (default: 42)
  --crop-cmd    crop command, called as: <crop-cmd> -i INPUT -o OUTPUT (default: sc_crop)

All images of a subject go to the same split, so no subject is in both train and validation.
The split is done within each dataset, so every dataset contributes ~val-size of its subjects.
Output files are named <dataset>_<original BIDS name>_crop.nii.gz, and the split is saved
to OUTPUT_DIR/split.csv so it can be reused without re-running the split.
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def read_file_list(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        dataset, rel = line.split("/", 1)
        sub = re.search(r"(?:^|/)sub-([^/_]+)", rel)
        if not sub:
            sys.exit(f"No subject found in: {line}")
        rows.append({"dataset": dataset, "subject": sub.group(1), "file": line})
    return pd.DataFrame(rows)


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--files", default=here / "files.txt", type=Path)
    ap.add_argument("--output", default=here / "cropped_data", type=Path)
    ap.add_argument("--val-size", default=0.2, type=float)
    ap.add_argument("--seed", default=42, type=int)
    ap.add_argument("--crop-cmd", default="sc_crop")
    args = ap.parse_args()

    df = read_file_list(args.files).sort_values("file").reset_index(drop=True)

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
        name = Path(r.file).name
        stem = name[:-len(".nii.gz")] if name.endswith(".nii.gz") else Path(name).stem
        ext = ".nii.gz" if name.endswith(".nii.gz") else Path(name).suffix
        return f"{r.split}/{r.dataset}_{stem}_crop{ext}"
    df["output"] = df.apply(out_name, axis=1)
    assert not df.output.duplicated().any(), "two inputs would get the same output name"

    args.output.mkdir(parents=True, exist_ok=True)
    df[["split", "dataset", "subject", "file", "output"]].to_csv(args.output / "split.csv", index=False)
    n_sub = groups.groupby(df.split).nunique()
    print(f"train: {(df.split == 'train').sum()} images / {n_sub.get('train', 0)} subjects | "
          f"validation: {(df.split == 'validation').sum()} images / {n_sub.get('validation', 0)} subjects")

    failed = []
    for r in df.itertuples():
        src = args.data_root / r.file
        dst = args.output / r.output
        if dst.exists():
            continue
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

    print(f"Done: {len(df) - len(failed)}/{len(df)} images in {args.output}")
    if failed:
        print(f"{len(failed)} failed:\n  " + "\n  ".join(failed))
        sys.exit(1)


if __name__ == "__main__":
    main()