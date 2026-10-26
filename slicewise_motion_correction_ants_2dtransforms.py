"""
Slicewise fMRI motion correction using ANTs (classical baseline).

Design:
1. Registers ACROSS TIMEPOINTS for each spatial slice, with 2D
   registration (matching the 2D-slicewise VoxelMorph model) — not
   a full 3D volume registration.
2. Uses a FIXED reference per slice (mean image across time, or the
   first timepoint) instead of chaining each timepoint to the
   previous one — this avoids drift/error accumulation along the
   time series.
3. Saves per-slice, per-timepoint motion parameters (rotation +
   translation) to a CSV, for QC/plotting.

Input is expected to be a 4D fMRI series (x, y, z, t).
"""

import csv
import os

import ants
import nibabel as nib
import numpy as np

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
input_path = "data/raw_data/fmri/ds004616/sub-02/ses-01/func/sub-02_ses-01_task-handgrasp_bold.nii.gz"
output_path = "motion_corrected_4d.nii.gz"
motion_params_csv = os.path.splitext(os.path.splitext(output_path)[0])[0] + "_motion_params.csv"
transform_type = "Rigid"      # Rigid recommended for inter-timepoint motion
reference_strategy = "mean"   # "mean" (recommended) or "first"


def decompose_rigid_2d(ants_transform):
    """Extract rotation (degrees) and translation (mm) from a 2D ANTs rigid transform."""
    params = np.array(ants_transform.parameters)
    matrix = params[:4].reshape(2, 2)
    translation_x, translation_y = params[4], params[5]
    rotation_deg = np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0]))
    return rotation_deg, translation_x, translation_y


# ----------------------------------------------------------------------
# 1. Load 4D fMRI data (x, y, z, t)
# ----------------------------------------------------------------------
nii = nib.load(input_path)
data = nii.get_fdata().astype(np.float32)
affine = nii.affine
zooms = nii.header.get_zooms()  # (dx, dy, dz, dt)

if data.ndim != 4:
    raise ValueError(f"Expected a 4D fMRI volume (x, y, z, t), got shape {data.shape}")

nx, ny, nz, nt = data.shape
in_plane_spacing = (float(zooms[0]), float(zooms[1]))

print(f"Loaded 4D volume: {data.shape} (x, y, z, t)")
print(f"In-plane spacing: {in_plane_spacing}")

# ----------------------------------------------------------------------
# 2. Slicewise (2D), per-timepoint registration to a FIXED reference
# ----------------------------------------------------------------------
corrected_data = np.zeros_like(data)
motion_records = []  # one dict per (slice, timepoint), written to CSV at the end

for z in range(nz):
    slice_series = data[:, :, z, :]  # shape (x, y, t)

    # --- Build the fixed 2D reference for this slice (computed once, never updated) ---
    if reference_strategy == "mean":
        reference_np = slice_series.mean(axis=2)
    elif reference_strategy == "first":
        reference_np = slice_series[:, :, 0]
    else:
        raise ValueError(f"Unknown reference_strategy: {reference_strategy}")

    fixed_slice = ants.from_numpy(reference_np, spacing=in_plane_spacing)

    for t in range(nt):
        moving_np = slice_series[:, :, t]

        # No need to register the reference timepoint to itself
        if reference_strategy == "first" and t == 0:
            corrected_data[:, :, z, t] = moving_np
            motion_records.append({
                "slice": z, "timepoint": t, "rotation_deg": 0.0,
                "translation_x_mm": 0.0, "translation_y_mm": 0.0,
                "transform_file": "",
            })
            continue

        moving_slice = ants.from_numpy(moving_np, spacing=in_plane_spacing)

        reg_result = ants.registration(
            fixed=fixed_slice,
            moving=moving_slice,
            type_of_transform=transform_type,
        )

        corrected_data[:, :, z, t] = reg_result["warpedmovout"].numpy()

        tx_file = reg_result["fwdtransforms"][0]
        tx = ants.read_transform(tx_file)
        rotation_deg, tx_mm, ty_mm = decompose_rigid_2d(tx)

        motion_records.append({
            "slice": z, "timepoint": t, "rotation_deg": rotation_deg,
            "translation_x_mm": tx_mm, "translation_y_mm": ty_mm,
            "transform_file": tx_file,
        })

    print(f"Slice {z + 1}/{nz} done")

# ----------------------------------------------------------------------
# 3. Save corrected 4D volume, preserving original affine/header
# ----------------------------------------------------------------------
corrected_img = nib.Nifti1Image(corrected_data, affine=affine, header=nii.header)
nib.save(corrected_img, output_path)
print(f"Motion-corrected volume saved to: {output_path}")

# ----------------------------------------------------------------------
# 4. Save per-slice, per-timepoint motion parameters to CSV
# ----------------------------------------------------------------------
fieldnames = ["slice", "timepoint", "rotation_deg", "translation_x_mm", "translation_y_mm", "transform_file"]
with open(motion_params_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(motion_records)

print(f"Motion parameters saved to: {motion_params_csv}")
