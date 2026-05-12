#!/usr/bin/env python3
"""
Dataset preparation for moco-dl with on-the-fly augmentation.

Differences from the original ivadomed/moco-dl/dataset_preparation.py
---------------------------------------------------------------------
- Saves the **clean** 4D volume under the key ``clean`` (not the
  pre-augmented ``aug_*.nii.gz``). On-the-fly augmentation in train.py
  produces the moving/GT pair every epoch.
- Splits are seeded for reproducibility.
- Existing .pt files are skipped on re-runs (idempotent).
- Test split: copies the raw, fixed, mask NIfTIs (and bval/bvec for dMRI)
  so test_model.py can read them; skipped if already present.
- Schema versioning: each .pt file includes ``schema_version="moco-dl-v2"``
  so future format changes can be detected.

Expected layout after preprocessing.py
--------------------------------------
    <base_dir>/
        sub-XX/[ses-YY/]
            dwi/
                sub-XX_dwi.nii.gz       (raw)
                dup_XX_fixed.nii.gz      (fixed reference, made by SCT)
                mask_XX.nii.gz            (spinal cord mask)
                ... (bval/bvec/json sidecars)
            func/
                sub-XX_bold.nii.gz       (raw)
                XX_mean.nii.gz            (mean reference)
                mask_XX.nii.gz            (spinal cord mask)

What this script writes
-----------------------
    <base_dir>/prepared/<mode>_dataset/
        training/<sub>/<subdir>/<sub>.pt
        validation/<sub>/<subdir>/<sub>.pt
        testing/<sub>/<subdir>/<original NIfTIs and sidecars>
        dataset.json

Usage
-----
    python dataset_preparation.py /path/to/data <dmri|fmri> [--seed 42]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from glob import glob
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

# We rely on the same config.yaml the rest of moco-dl uses, but loaded
# without depending on the original config_loader (so this script is
# self-contained inside our refactor).
import yaml


SCHEMA_VERSION = "moco-dl-v2"  # bump if the .pt schema changes


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(config_path: Path | None = None) -> dict:
    """Load file-pattern config. Falls back to the values from the moco-dl repo."""
    if config_path is not None and config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    # Default mirrors moco-dl/config.yaml
    return {
        "dmri": {
            "raw":    "sub-*_dwi.nii.gz",
            "fixed":  "dup_*_fixed.nii.gz",
            "mask":   "mask_*.nii.gz",
            "subdir": "dwi",
            "suffix": "dwi",
        },
        "fmri": {
            "raw":    "sub-*_bold.nii.gz",
            "fixed":  "*_mean.nii.gz",
            "mask":   "mask_*.nii.gz",
            "subdir": "func",
            "suffix": "bold",
        },
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_as_tensor(nii_path: Path) -> tuple[torch.Tensor, np.ndarray]:
    """Load NIfTI -> float32 tensor with a leading channel axis, plus its affine."""
    img = nib.load(str(nii_path))
    data = img.get_fdata().astype(np.float32)
    tensor = torch.from_numpy(data).unsqueeze(0)  # (1, ...)
    return tensor, img.affine.astype(np.float32)


def find_one(folder: Path, pattern: str) -> Path | None:
    """Return the first match for ``pattern`` in ``folder``, or None."""
    matches = sorted(folder.glob(pattern))
    return matches[0] if matches else None


def collect_subjects(base_dir: Path) -> list[str]:
    """
    Enumerate subject directories. If a subject has session folders, list
    each session as ``sub-XX_ses-YY``; otherwise just ``sub-XX``.
    """
    subjects: list[str] = []
    for entry in sorted(base_dir.iterdir()):
        if not entry.is_dir() or not entry.name.startswith("sub-"):
            continue
        sessions = sorted(d.name for d in entry.iterdir()
                          if d.is_dir() and d.name.startswith("ses-"))
        if sessions:
            for ses in sessions:
                subjects.append(f"{entry.name}_{ses}")
        else:
            subjects.append(entry.name)
    return subjects


def split_subjects(subjects: list[str], seed: int,
                   train_ratio: float = 0.8, val_ratio: float = 0.1
                   ) -> tuple[list[str], list[str], list[str]]:
    """
    Deterministic 80/10/10 split with a fixed seed.

    Edge case: if rounding would leave validation or testing empty when
    we have at least 3 subjects, take one subject from training and give
    it to whichever split is empty. This avoids breaking train.py's
    EarlyStopping (which needs validation data).
    """
    rng = random.Random(seed)
    shuffled = list(subjects)
    rng.shuffle(shuffled)
    n = len(shuffled)
    if n == 0:
        return [], [], []

    n_train = round(train_ratio * n)
    n_val = round(val_ratio * n)
    # Ensure at least 1 in val when we have ≥3 subjects
    if n_val == 0 and n >= 3 and n_train >= 2:
        n_val = 1
        n_train -= 1
    n_test = n - n_train - n_val
    # Ensure at least 1 in test when we have ≥3 subjects
    if n_test == 0 and n >= 3 and n_train >= 2:
        n_test = 1
        n_train -= 1

    return (shuffled[:n_train],
            shuffled[n_train:n_train + n_val],
            shuffled[n_train + n_val:n_train + n_val + n_test])


def resolve_subject_folder(base_dir: Path, sub_or_ses: str, subdir: str) -> Path:
    """Get the data folder (.../dwi or .../func) for a subject or session."""
    if "_ses-" in sub_or_ses:
        sub, ses = sub_or_ses.split("_", 1)
        return base_dir / sub / ses / subdir
    return base_dir / sub_or_ses / subdir


# ---------------------------------------------------------------------------
# Per-subject conversion
# ---------------------------------------------------------------------------

def make_pt_for_subject(sub: str, base_dir: Path, mode_dir: Path,
                        split: str, patterns: dict, force: bool = False) -> dict | None:
    """
    Build a single .pt file for a (training/validation) subject. Returns the
    JSON entry (with relative path), or None if files are missing.
    """
    data_folder = resolve_subject_folder(base_dir, sub, patterns["subdir"])
    if not data_folder.exists():
        print(f"  [skip] {sub}: folder not found at {data_folder}")
        return None

    raw = find_one(data_folder, patterns["raw"])
    fixed = find_one(data_folder, patterns["fixed"])
    mask = find_one(data_folder, patterns["mask"])
    missing = [name for name, p in [("raw", raw), ("fixed", fixed), ("mask", mask)] if p is None]
    if missing:
        print(f"  [skip] {sub}: missing {missing} (looked in {data_folder})")
        return None

    out_folder = mode_dir / split / sub / patterns["subdir"]
    out_folder.mkdir(parents=True, exist_ok=True)
    pt_name = raw.name.replace(".nii.gz", ".pt")
    pt_path = out_folder / pt_name

    if pt_path.exists() and not force:
        rel = pt_path.relative_to(mode_dir)
        return {"data": str(rel)}

    # Load and save
    clean, affine = load_as_tensor(raw)
    fixed_t, _ = load_as_tensor(fixed)
    mask_t, _ = load_as_tensor(mask)

    torch.save({
        "clean":          clean,
        "fixed":          fixed_t,
        "mask":           mask_t,
        "affine":         torch.from_numpy(affine),
        "schema_version": SCHEMA_VERSION,
        "subject":        sub,
        "source_raw":     str(raw),
    }, pt_path)
    print(f"  [saved] {pt_path.relative_to(mode_dir)}  "
          f"(clean shape {tuple(clean.shape)})")

    rel = pt_path.relative_to(mode_dir)
    return {"data": str(rel)}


def copy_test_files(sub: str, base_dir: Path, mode_dir: Path,
                    patterns: dict, mode: str) -> dict | None:
    """
    Copy raw/fixed/mask (and bval/bvec for dMRI) to the testing folder so
    test_model.py can read them. Returns a JSON entry pointing at the
    copied files (relative paths), or None if anything is missing.
    """
    data_folder = resolve_subject_folder(base_dir, sub, patterns["subdir"])
    if not data_folder.exists():
        print(f"  [skip] {sub}: folder not found at {data_folder}")
        return None

    raw = find_one(data_folder, patterns["raw"])
    fixed = find_one(data_folder, patterns["fixed"])
    mask = find_one(data_folder, patterns["mask"])
    if raw is None or fixed is None or mask is None:
        print(f"  [skip-test] {sub}: missing one of raw/fixed/mask")
        return None

    out_folder = mode_dir / "testing" / sub / patterns["subdir"]
    out_folder.mkdir(parents=True, exist_ok=True)

    copies = [raw, fixed, mask]
    if mode == "dmri":
        copies += list(data_folder.glob("*.bval"))
        copies += list(data_folder.glob("*.bvec"))

    for src in copies:
        dst = out_folder / src.name
        if not dst.exists():
            shutil.copy2(src, dst)

    rel_subdir = (mode_dir / "testing" / sub / patterns["subdir"]).relative_to(mode_dir)
    return {
        "raw":   str(rel_subdir / raw.name),
        "fixed": str(rel_subdir / fixed.name),
        "mask":  str(rel_subdir / mask.name),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("base_dir", type=Path,
                   help="Root directory containing sub-* folders (already preprocessed by SCT).")
    p.add_argument("mode", choices=["dmri", "fmri"])
    p.add_argument("--seed", type=int, default=42,
                   help="Seed for the train/val/test split.")
    p.add_argument("--config", type=Path, default=None,
                   help="Path to a config.yaml. Defaults to the values from moco-dl/config.yaml.")
    p.add_argument("--train-ratio", type=float, default=0.80)
    p.add_argument("--val-ratio", type=float, default=0.10)
    p.add_argument("--force", action="store_true",
                   help="Re-convert even if .pt files already exist.")
    p.add_argument("--output", type=Path, default=None,
                   help="Output directory (default: <base_dir>/prepared)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.mode not in cfg:
        sys.exit(f"ERROR: mode '{args.mode}' not in config")
    patterns = cfg[args.mode]

    out_root = args.output if args.output is not None else args.base_dir / "prepared"
    mode_dir = out_root / f"{args.mode}_dataset"
    mode_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {mode_dir}")

    # Subject discovery + split
    subjects = collect_subjects(args.base_dir)
    if not subjects:
        sys.exit(f"ERROR: no sub-* directories found in {args.base_dir}")
    train, val, test = split_subjects(subjects, args.seed,
                                      args.train_ratio, args.val_ratio)
    print(f"Found {len(subjects)} subjects/sessions  "
          f"-> train={len(train)}, val={len(val)}, test={len(test)}  "
          f"(seed={args.seed})")

    # Build training and validation .pt files
    splits: dict[str, list[dict]] = {"training": [], "validation": [], "testing": []}
    for split_name, subj_list in [("training", train), ("validation", val)]:
        print(f"\n--- {split_name} ---")
        for sub in subj_list:
            entry = make_pt_for_subject(sub, args.base_dir, mode_dir,
                                        split_name, patterns, force=args.force)
            if entry is not None:
                splits[split_name].append(entry)

    # Copy test files
    print(f"\n--- testing ---")
    for sub in test:
        entry = copy_test_files(sub, args.base_dir, mode_dir, patterns, args.mode)
        if entry is not None:
            splits["testing"].append(entry)

    # Write dataset.json
    dataset_json = mode_dir / "dataset.json"
    with open(dataset_json, "w") as f:
        json.dump({
            "schema_version": SCHEMA_VERSION,
            "mode":           args.mode,
            "seed":           args.seed,
            "training":       splits["training"],
            "validation":     splits["validation"],
            "testing":        splits["testing"],
        }, f, indent=2)

    print(f"\nWrote {dataset_json}")
    print(f"  training:   {len(splits['training'])}")
    print(f"  validation: {len(splits['validation'])}")
    print(f"  testing:    {len(splits['testing'])}")


if __name__ == "__main__":
    main()
