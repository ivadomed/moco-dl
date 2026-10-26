# -*- coding: utf-8 -*-
"""
Pre-processing script used to prepare the necessary files/data for motion correction training,
separately handling dMRI and fMRI using functions from the Spinal Cord Toolbox (SCT).

Pipeline Overview:
------------------
For dMRI data:
    - Separate b0 and dwi volumes to get mean b0 and mean dwi
    - Perform spinal cord segmentation
    - Create mask around the spinal cord
    - Duplicate mean b0 and mean dwi volumes into a 4D fixed reference (H, W, D, T)

For fMRI data:
    - Compute mean across time to get mean fmri
    - Perform spinal cord segmentation
    - Create mask around the spinal cord

Usage:
------
In terminal/command line:

    python preprocessing.py /path/to/data mode

Arguments:
    /path/to/data : data directory for training dataset preparation
    mode          : 'dmri' or 'fmri' depending on dataset type
"""

import os
import sys
import glob
import subprocess
import numpy as np
import nibabel as nib

from config_loader import config

def process_subject(target, mode):
    print(f"\nProcessing {target} in {mode.upper()} mode ...")

    patterns = config[mode]
    subdir = patterns["subdir"]

    # ---------------------------
    # Find raw NIfTI
    # ---------------------------
    nii_files = glob.glob(os.path.join(target, subdir, patterns["raw"]))
    if not nii_files:
        print(f"No {mode} NIfTI found in {target}")
        return
    input_img = nii_files[0]
    nii = nib.load(input_img)
    T = nii.shape[3]
    print("Found NIfTI:", input_img)

    # prefix (sub or sub+ses)
    root = os.path.basename(input_img)
    parts = root.split("_")
    if len(parts) > 1 and parts[1].startswith("ses-"):
        prefix = "_".join(parts[:2])
    else:
        prefix = parts[0]

    if mode == "dmri":
        # ---------------------------
        # Find bvec + bval
        # ---------------------------
        bvec_files = glob.glob(os.path.join(target, subdir, "*.bvec"))
        if not bvec_files:
            print(f"No bvec file found in {target}")
            return
        bvec_file = bvec_files[0]

        bval_files = glob.glob(os.path.join(target, subdir, "*.bval"))
        if not bval_files:
            print(f"No bval file found in {target}")
            return
        bval_file = bval_files[0]

        out_dir = os.path.join(target, subdir)

        # ---------------------------
        # Separate b0 and dwi to get mean dwi
        # ---------------------------
        SCT_SEP = [
            "sct_dmri_separate_b0_and_dwi",
            "-i", input_img,
            "-bvec", bvec_file,
            "-bval", bval_file,
            "-ofolder", out_dir
        ]
        subprocess.run(SCT_SEP, check=True)
        print("Separation Success!")

        # mean dwi
        nii_files = glob.glob(os.path.join(out_dir, "*dwi_mean.nii.gz"))
        if not nii_files:
            print(f"No mean dwi found in {target}")
            return
        mean_img = nii_files[0]
        print("Found mean dwi:", mean_img)

        # ---------------------------
        # Segmentation spinal cord
        # ---------------------------
        SCT_SEG = [
            "sct_deepseg",
            "spinalcord",
            "-i", mean_img,
        ]
        subprocess.run(SCT_SEG, check=True)
        print("Segmentation Success!")

        # segmentation output
        seg_files = glob.glob(os.path.join(os.path.dirname(mean_img), "*seg.nii.gz"))
        if not seg_files:
            print(f"No seg file produced for {target}")
            return
        seg_img = seg_files[0]

        # ---------------------------
        # Create MASK along the spinal cord
        # ---------------------------
        SCT_MASK = [
            "sct_create_mask",
            "-i", mean_img,
            "-p", f"centerline,{seg_img}",
            "-size", "35mm"
        ]
        subprocess.run(SCT_MASK, check=True, cwd=out_dir)
        print("Mask Success!")

    elif mode == "fmri":
        out_dir = os.path.join(target, subdir)
        mean_img = os.path.join(out_dir, f"{prefix}_fmri_mean.nii.gz")

        # ---------------------------
        # Mean fMRI across time
        # ---------------------------
        SCT_MATH = [
            "sct_maths",
            "-i", input_img,
            "-mean", "t",
            "-o", mean_img
        ]
        subprocess.run(SCT_MATH, check=True, cwd=out_dir)
        print("Mean Success!")

        seg_img = os.path.join(out_dir, f"{prefix}_seg.nii.gz")
        # ---------------------------
        # Segmentation spinal cord
        # ---------------------------
        SCT_SEG = [
            "sct_deepseg",
            "sc_epi",
            "-i", mean_img,
            "-o", seg_img
        ]
        subprocess.run(SCT_SEG, check=True)
        print("Segmentation Success!")

        # segmentation output
        seg_files = glob.glob(os.path.join(os.path.dirname(mean_img), "*seg.nii.gz"))
        if not seg_files:
            print(f"No seg file produced for {target}")
            return
        seg_img = seg_files[0]

        # ---------------------------
        # Create MASK along the spinal cord
        # ---------------------------
        SCT_MASK = [
            "sct_maths",
            "-i", seg_img,
            "-dilate", "15",
            "-shape", "disk",
            "-o", f"mask_{prefix}.nii.gz",
            "-dim", "2"
        ]
        subprocess.run(SCT_MASK, check=True, cwd=out_dir)
        print("Mask Success!")

    # ---------------------------
    # Create fixed volume reference --> 4D for dMRI, 3D for fMRI
    # ---------------------------
    if mode == "dmri":
        nii_files = glob.glob(os.path.join(target, subdir, "*b0_mean.nii.gz"))
        if not nii_files:
            print(f"No mean b0 found in {target}")
            return
        mean_b0_img = nii_files[0]

        data = nii.get_fdata()
        affine = nii.affine
        header = nii.header

        mean_b0_data = nib.load(mean_b0_img).get_fdata()
        mean_dwi_data = nib.load(mean_img).get_fdata()
        bvals = np.loadtxt(bval_file)

        # ---------------------------
        # Duplicate along time point (mean b0 where b0s are and mean dwi for the others)
        # ---------------------------
        corrected = np.zeros_like(data)
        for t in range(T):
            if bvals[t] < 50:  # b0
                corrected[..., t] = mean_b0_data
            else:  # dwi
                corrected[..., t] = mean_dwi_data

        out_corrected = os.path.join(target, subdir, f"dup_{prefix}_fixed.nii.gz")
        nib.save(nib.Nifti1Image(corrected, affine, header), out_corrected)
        print("Saved corrected file:", out_corrected)

    elif mode == "fmri":
        pass # no duplication for fMRI

# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python preprocessing.py <data_dir> <dmri|fmri>")
        sys.exit(1)

    data_dir = sys.argv[1]
    mode = sys.argv[2].lower()
    subfolders = sorted([f.path for f in os.scandir(data_dir) if f.is_dir()])

    for sub in subfolders:
        # session-aware
        sessions = sorted([f.path for f in os.scandir(sub) if f.is_dir() and f.name.startswith("ses-")])
        targets = sessions if sessions else [sub]

        for target in targets:
            process_subject(target, mode)