# -*- coding: utf-8 -*-
"""
Augmentation script to simulate motion artifacts in dMRI and fMRI data by applying slice-wise affine transformations.

- Slice-wise in-plane translations (x-y motion)
- Compatible with both dMRI and fMRI (user specifies mode)
- Motion realism controls:
    - Case-level probability: some subjects have no motion
    - Timepoint-level probability: only some volumes include motion
    - Slice-level probability: only a subset of slices move within a moving volume
- Preserves anatomical integrity while introducing controlled synthetic motion

Usage:
------
In terminal/command line:

    python augmentation.py /path/to/data mode

Arguments:
    /path/to/data : data directory for training dataset preparation
    mode          : 'dmri' or 'fmri' depending on dataset type
"""

import os
import sys
import glob
import nibabel as nib
import numpy as np

from config_loader import config
from scipy.ndimage import affine_transform

# ---------------------------
# Custom slice-wise transform
# ---------------------------
class RandSliceWiseAffine:
    """
    Apply random slice-wise affine transform (shift) to a 3D volume.
    - p_case: probability dataset has any motion
    - p_frame: probability timepoint has motion
    - p_slice: probability slice moves in a moving timepoint
    """
    def __init__(self, max_shift=2, p_case=0.9, p_frame=0.6, p_slice=0.4, axis=2):
        self.max_shift = max_shift
        self.p_case = p_case
        self.p_frame = p_frame
        self.p_slice = p_slice
        self.axis = axis
        self.case_has_motion = True  # placeholder; set per case

    def new_case(self):
        """Resample whether THIS case has any motion."""
        self.case_has_motion = (np.random.rand() < self.p_case)

    def __call__(self, vol: np.ndarray) -> np.ndarray:
        out = vol.copy()

        # If this case has NO motion at all
        if not self.case_has_motion:
            return out

        # Per-frame decision
        if np.random.rand() > self.p_frame:
            return out

        # Frame has motion → slice selections
        if self.axis != 2:
            out = np.moveaxis(out, self.axis, -1)

        H, W, D = out.shape
        for d in range(D):
            if np.random.rand() < self.p_slice:
                tx = np.random.uniform(-self.max_shift, self.max_shift)
                ty = np.random.uniform(-self.max_shift, self.max_shift)
                out[:, :, d] = affine_transform(
                    out[:, :, d], np.eye(2), offset=(tx, ty), order=0, mode="nearest"
                )

        if self.axis != 2:
            out = np.moveaxis(out, -1, self.axis)
        return out

# ---------------------------
# Augmentation functions
# ---------------------------
def augment_data(input_img, output_img, slicewise_tf):
    """
    Load a 4D dataset and apply slice-wise augmentation volume-by-volume.
    """
    print(f"Augmenting: {input_img}")
    img = nib.load(input_img)
    data = img.get_fdata().astype(np.float32)
    slicewise_tf.new_case()
    H, W, D, T = data.shape
    print("Input shape:", data.shape)

    aug_data = np.zeros_like(data)
    for t in range(T):
        aug_data[:, :, :, t] = slicewise_tf(data[:, :, :, t])
        print(f"Augmented volume {t + 1}/{T}")

    aug_img = nib.Nifti1Image(aug_data, img.affine, img.header)
    nib.save(aug_img, output_img)
    print(f"Saved augmented dataset: {output_img}")

# ---------------------------
# Main
# ---------------------------
def main(data_dir, mode):
    """
    Perform augmentation for all subjects in dataset directory (dmri or fmri).
    """
    slicewise_tf = RandSliceWiseAffine(max_shift=2, p_case=0.8, p_frame=0.8, p_slice=0.8, axis=2)
    subfolders = sorted([f.path for f in os.scandir(data_dir) if f.is_dir()])

    patterns = config[mode]

    for sub in subfolders:
        # handle sessions if they exist
        sessions = sorted([f.path for f in os.scandir(sub) if f.is_dir() and f.name.startswith("ses-")])
        targets = sessions if sessions else [sub]

        for target in targets:
            print(f"\nProcessing {target} ...")

            # use the config-defined subdir + raw pattern
            nii_files = glob.glob(os.path.join(target, patterns["subdir"], patterns["raw"]))
            if not nii_files:
                print(f"No {mode} NIfTI found in {target}")
                continue
            input_img = nii_files[0]

            # construct prefix
            fname = os.path.basename(input_img)
            parts = fname.split("_")
            if len(parts) > 1 and parts[1].startswith("ses-"):
                prefix = "_".join(parts[:2])
            else:
                prefix = parts[0]

            # build output path
            output_img = os.path.join(
                target, patterns["subdir"], f"aug_{prefix}_{patterns['suffix']}.nii.gz"
            )

            # run augmentation
            augment_data(input_img, output_img, slicewise_tf)

# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python augmentation.py <data_dir> <dmri|fmri>")
        sys.exit(1)

    data_dir = sys.argv[1]
    mode = sys.argv[2].lower()
    main(data_dir, mode)
