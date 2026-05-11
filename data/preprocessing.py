# -*- coding: utf-8 -*-
"""
Pre-processing script for moco-dl: prepares the SCT-derived files
(mean reference, cord segmentation, mask) needed for training.

Differences from the original
-----------------------------
- BIDS-aware subject discovery: only descends into 'sub-*' folders, ignoring
  '.git', 'derivatives', 'sourcedata', 'code', etc.
- Canonical-file selection for dMRI: prefers 'sub-XX_dwi.nii.gz' over
  'sub-XX_acq-b0_dwi.nii.gz' or 'sub-XX_rec-average_dwi.nii.gz', and verifies
  that the number of volumes matches the bval/bvec entries before running
  SCT. This avoids the crash on subjects with multiple dwi variants.
- Picks the dwi-vs-b0 mean files explicitly (no fragile glob like
  '*dwi_mean') so re-runs don't pick up the wrong file.
- Idempotent: if all expected outputs already exist for a subject, the
  subject is skipped (use --force to re-run).
- Continues on per-subject errors (a single bad subject doesn't abort
  the entire run); a summary is printed at the end.

Pipeline (unchanged in spirit)
------------------------------
For dMRI: separate b0/dwi -> segment cord on mean dwi -> create cord mask ->
   build 4D 'fixed' volume by duplicating mean-b0 / mean-dwi per timepoint.
For fMRI: temporal mean -> segment cord on mean -> create cord mask.

Output filenames (match the moco-dl config.yaml so dataset_preparation.py works):
   <sub>_dwi.nii.gz                       (original raw, untouched)
   <sub>_dwi_b0_mean.nii.gz               (SCT output)
   <sub>_dwi_dwi_mean.nii.gz              (SCT output)
   <sub>_dwi_dwi_mean_seg.nii.gz          (SCT output)
   mask_<sub>_dwi_dwi_mean.nii.gz         (dMRI mask)
   dup_<sub>_fixed.nii.gz                 (dMRI 'fixed')
   <sub>_fmri_mean.nii.gz                 (fMRI mean reference)
   <sub>_seg.nii.gz                       (fMRI cord seg)
   mask_<sub>.nii.gz                      (fMRI mask)

Usage
-----
   python preprocessing.py /path/to/data <dmri|fmri> [--force] [--stop-on-error]
"""

import argparse
import glob
import os
import subprocess
import sys
import traceback

import nibabel as nib
import numpy as np

from config_loader import config


# ---------------------------------------------------------------------------
# Subject discovery
# ---------------------------------------------------------------------------

def is_subject_dir(path: str) -> bool:
    return os.path.isdir(path) and os.path.basename(path).startswith("sub-")


def is_session_dir(path: str) -> bool:
    return os.path.isdir(path) and os.path.basename(path).startswith("ses-")


def enumerate_targets(data_dir: str) -> list:
    """Return every (sub or sub/ses) directory under data_dir, sorted."""
    targets = []
    for sub in sorted(os.scandir(data_dir), key=lambda e: e.name):
        if not is_subject_dir(sub.path):
            continue
        sessions = sorted([s.path for s in os.scandir(sub.path) if is_session_dir(s.path)])
        if sessions:
            targets.extend(sessions)
        else:
            targets.append(sub.path)
    return targets


# ---------------------------------------------------------------------------
# Canonical-file selection
# ---------------------------------------------------------------------------

# Filenames containing any of these BIDS entity tags are *derivative*
# acquisitions and should not be the primary file for processing.
_DERIVATIVE_TAGS = ("acq-", "rec-", "run-", "_part-", "_dir-")


def pick_canonical_nifti(folder: str, suffix: str) -> str:
    """
    Find the canonical raw NIfTI in `folder` matching '*_<suffix>.nii.gz'.

    Strategy: prefer files with no acquisition-variant tags in their name.
    Among matches, prefer the shortest filename (fewest BIDS entities =
    most canonical). Returns None if no canonical file is present.
    """
    candidates = glob.glob(os.path.join(folder, "*_{}.nii.gz".format(suffix)))
    if not candidates:
        return None
    canonical = [c for c in candidates
                 if not any(t in os.path.basename(c) for t in _DERIVATIVE_TAGS)]
    if canonical:
        return sorted(canonical, key=lambda p: len(os.path.basename(p)))[0]
    return None


def _count_volumes(nifti_path: str) -> int:
    img = nib.load(nifti_path)
    return img.shape[3] if img.ndim >= 4 else 1


def _count_bval_entries(bval_path: str) -> int:
    with open(bval_path) as f:
        return len(f.read().split())


def validate_dwi_matches_bvals(dwi_path: str, bval_path: str):
    """Confirm the dwi NIfTI and bval file describe the same number of volumes."""
    try:
        n_vol = _count_volumes(dwi_path)
        n_bval = _count_bval_entries(bval_path)
    except Exception as e:
        return False, "could not read shapes: {}".format(e)
    if n_vol != n_bval:
        return False, "dwi has {} volumes but bval has {} entries".format(n_vol, n_bval)
    return True, ""


def derive_prefix(input_nii: str) -> str:
    """sub-XX  ->  'sub-XX' ;  sub-XX_ses-YY_*  ->  'sub-XX_ses-YY'."""
    root = os.path.basename(input_nii)
    parts = root.split("_")
    if len(parts) > 1 and parts[1].startswith("ses-"):
        return "_".join(parts[:2])
    return parts[0]


# ---------------------------------------------------------------------------
# Idempotency checks
# ---------------------------------------------------------------------------

def _have_dmri_outputs(out_dir: str, prefix: str) -> bool:
    required = [
        os.path.join(out_dir, "{}_dwi_dwi_mean.nii.gz".format(prefix)),
        os.path.join(out_dir, "{}_dwi_b0_mean.nii.gz".format(prefix)),
        os.path.join(out_dir, "{}_dwi_dwi_mean_seg.nii.gz".format(prefix)),
        os.path.join(out_dir, "dup_{}_fixed.nii.gz".format(prefix)),
    ]
    mask_match = glob.glob(os.path.join(out_dir, "mask_*.nii.gz"))
    return all(os.path.exists(p) for p in required) and bool(mask_match)


def _have_fmri_outputs(out_dir: str, prefix: str) -> bool:
    return (os.path.exists(os.path.join(out_dir, "{}_fmri_mean.nii.gz".format(prefix)))
            and os.path.exists(os.path.join(out_dir, "{}_seg.nii.gz".format(prefix)))
            and os.path.exists(os.path.join(out_dir, "mask_{}.nii.gz".format(prefix))))


# ---------------------------------------------------------------------------
# Per-subject processing
# ---------------------------------------------------------------------------

def process_subject(target: str, mode: str, force: bool = False) -> str:
    """
    Process one (sub or sub/ses) directory. Returns:
      'ok'      -> processed cleanly
      'skipped' -> outputs already present
      'failed: <why>' -> something went wrong
    """
    print("\nProcessing {} in {} mode ...".format(target, mode.upper()))
    patterns = config[mode]
    subdir = patterns["subdir"]
    folder = os.path.join(target, subdir)
    if not os.path.isdir(folder):
        return "failed: missing {}/ folder".format(subdir)

    suffix = patterns["suffix"]  # 'dwi' or 'bold'
    input_img = pick_canonical_nifti(folder, suffix)
    if input_img is None:
        print("  -> no canonical {} NIfTI in {}".format(suffix, folder))
        return "failed: no canonical NIfTI"
    print("  Canonical input: {}".format(input_img))

    prefix = derive_prefix(input_img)
    out_dir = folder

    if not force:
        if mode == "dmri" and _have_dmri_outputs(out_dir, prefix):
            print("  -> outputs already present, skipping (use --force to re-run)")
            return "skipped"
        if mode == "fmri" and _have_fmri_outputs(out_dir, prefix):
            print("  -> outputs already present, skipping (use --force to re-run)")
            return "skipped"

    # ----------------- dMRI -----------------
    if mode == "dmri":
        all_bvec = sorted(glob.glob(os.path.join(folder, "{}*.bvec".format(prefix))))
        all_bval = sorted(glob.glob(os.path.join(folder, "{}*.bval".format(prefix))))
        if not all_bvec or not all_bval:
            return "failed: missing bvec/bval"
        bvec_canon = [b for b in all_bvec
                      if not any(t in os.path.basename(b) for t in _DERIVATIVE_TAGS)]
        bval_canon = [b for b in all_bval
                      if not any(t in os.path.basename(b) for t in _DERIVATIVE_TAGS)]
        bvec_file = (bvec_canon or all_bvec)[0]
        bval_file = (bval_canon or all_bval)[0]

        ok, why = validate_dwi_matches_bvals(input_img, bval_file)
        if not ok:
            print("  -> volume/bval mismatch: {}".format(why))
            return "failed: volume/bval mismatch: {}".format(why)

        subprocess.run([
            "sct_dmri_separate_b0_and_dwi",
            "-i", input_img,
            "-bvec", bvec_file,
            "-bval", bval_file,
            "-ofolder", out_dir,
        ], check=True)
        print("  Separation done")

        mean_dwi = os.path.join(out_dir, "{}_dwi_dwi_mean.nii.gz".format(prefix))
        mean_b0 = os.path.join(out_dir, "{}_dwi_b0_mean.nii.gz".format(prefix))
        if not (os.path.exists(mean_dwi) and os.path.exists(mean_b0)):
            return "failed: SCT did not produce expected mean files"

        subprocess.run([
            "sct_deepseg", "spinalcord", "-i", mean_dwi,
        ], check=True)
        print("  Segmentation done")

        seg_img = os.path.join(out_dir, "{}_dwi_dwi_mean_seg.nii.gz".format(prefix))
        if not os.path.exists(seg_img):
            return "failed: missing seg output {}".format(seg_img)

        subprocess.run([
            "sct_create_mask",
            "-i", mean_dwi,
            "-p", "centerline,{}".format(seg_img),
            "-size", "35mm",
        ], check=True, cwd=out_dir)
        print("  Mask done")

        # Build 4D 'fixed' reference: duplicate mean_b0 / mean_dwi per timepoint
        nii = nib.load(input_img)
        T = nii.shape[3]
        bvals = np.loadtxt(bval_file)
        mean_b0_data = nib.load(mean_b0).get_fdata()
        mean_dwi_data = nib.load(mean_dwi).get_fdata()

        corrected = np.zeros_like(nii.get_fdata())
        for t in range(T):
            corrected[..., t] = mean_b0_data if bvals[t] < 50 else mean_dwi_data

        out_corrected = os.path.join(out_dir, "dup_{}_fixed.nii.gz".format(prefix))
        nib.save(nib.Nifti1Image(corrected, nii.affine, nii.header), out_corrected)
        print("  Wrote {}".format(out_corrected))
        return "ok"

    # ----------------- fMRI -----------------
    if mode == "fmri":
        mean_img = os.path.join(out_dir, "{}_fmri_mean.nii.gz".format(prefix))
        subprocess.run([
            "sct_maths", "-i", input_img, "-mean", "t", "-o", mean_img,
        ], check=True)
        print("  Mean done")

        seg_img = os.path.join(out_dir, "{}_seg.nii.gz".format(prefix))
        subprocess.run([
            "sct_deepseg", "sc_epi", "-i", mean_img, "-o", seg_img,
        ], check=True)
        print("  Segmentation done")

        if not os.path.exists(seg_img):
            return "failed: missing seg output {}".format(seg_img)

        subprocess.run([
            "sct_maths",
            "-i", seg_img,
            "-dilate", "15",
            "-shape", "disk",
            "-o", "mask_{}.nii.gz".format(prefix),
            "-dim", "2",
        ], check=True, cwd=out_dir)
        print("  Mask done")
        return "ok"

    return "failed: unknown mode {}".format(mode)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("data_dir", help="Root directory containing sub-* folders.")
    p.add_argument("mode", choices=["dmri", "fmri"])
    p.add_argument("--force", action="store_true",
                   help="Re-run SCT even when outputs already exist.")
    p.add_argument("--stop-on-error", action="store_true",
                   help="Abort the run on the first failing subject "
                        "(default: log and continue).")
    return p.parse_args()


def main():
    args = parse_args()
    targets = enumerate_targets(args.data_dir)
    if not targets:
        sys.exit("No sub-* directories found in {}".format(args.data_dir))

    print("Found {} target(s) under {}".format(len(targets), args.data_dir))
    summary = {"ok": [], "skipped": [], "failed": []}

    for target in targets:
        try:
            status = process_subject(target, args.mode, force=args.force)
        except subprocess.CalledProcessError as e:
            status = "failed: SCT returncode {}".format(e.returncode)
            if args.stop_on_error:
                raise
        except Exception as e:
            status = "failed: {}: {}".format(type(e).__name__, e)
            traceback.print_exc()
            if args.stop_on_error:
                raise

        if status == "ok":
            summary["ok"].append(target)
        elif status == "skipped":
            summary["skipped"].append(target)
        else:
            summary["failed"].append((target, status))

    print("\n" + "=" * 70)
    print("Summary ({}):".format(args.mode))
    print("  ok      : {}".format(len(summary["ok"])))
    print("  skipped : {}".format(len(summary["skipped"])))
    print("  failed  : {}".format(len(summary["failed"])))
    if summary["failed"]:
        print("\nFailed subjects:")
        for target, why in summary["failed"]:
            print("  - {}   [{}]".format(target, why))


if __name__ == "__main__":
    main()
