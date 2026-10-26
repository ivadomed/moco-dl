"""
Slicewise sequential motion correction for spinal cord fMRI using VoxelMorph.

Pipeline:
- Load 4D NIfTI (X, Y, Z, T) fMRI volumes, one per subject
- Build sequential pairs per slice: (moving=slice_t, fixed=slice_{t-1})
- Train VxmDense (2D) with NCC similarity + smoothness regularization
- No segmentation used (raw images only, per current project decision)

Requires: pip install voxelmorph nibabel torch
"""

import os
os.environ['VXM_BACKEND'] = 'pytorch'

import glob
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
import voxelmorph as vxm

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
DATA_DIR = "data"   # expects one *.nii.gz per subject, 4D (X, Y, Z, T)
TARGET_SHAPE = (128, 128)              # fixed in-plane size after crop/pad
BATCH_SIZE = 16
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4
LAMBDA_SMOOTH = 1.0
NCC_WINDOW = [9, 9]
INT_STEPS = 7                          # diffeomorphic integration steps; set 0 to disable
VAL_FRACTION = 0.2                     # held out by subject
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = "claude/fmri_moco/checkpoints"


# --------------------------------------------------------------------------
# Preprocessing helpers
# --------------------------------------------------------------------------
def normalize(slice_2d: np.ndarray) -> np.ndarray:
    """Percentile-clipped min-max normalization, robust to outlier voxels."""
    slice_2d = slice_2d.astype(np.float32)
    p1, p99 = np.percentile(slice_2d, (1, 99))
    slice_2d = np.clip(slice_2d, p1, p99)
    denom = (p99 - p1) if (p99 - p1) > 1e-8 else 1e-8
    return (slice_2d - p1) / denom


def pad_or_crop(slice_2d: np.ndarray, target_shape=TARGET_SHAPE) -> np.ndarray:
    """Center pad (if smaller) or center crop (if larger) to target_shape."""
    h, w = slice_2d.shape
    th, tw = target_shape

    # pad if needed
    pad_h, pad_w = max(th - h, 0), max(tw - w, 0)
    if pad_h > 0 or pad_w > 0:
        slice_2d = np.pad(
            slice_2d,
            ((pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2)),
            mode="constant",
        )

    # crop if needed (center crop)
    h, w = slice_2d.shape
    start_h = max((h - th) // 2, 0)
    start_w = max((w - tw) // 2, 0)
    return slice_2d[start_h:start_h + th, start_w:start_w + tw]


# --------------------------------------------------------------------------
# Pair construction
# --------------------------------------------------------------------------
def build_pairs_for_subject(nifti_path: str, subject_id: str):
    """Build sequential (t, t-1) pairs for every slice of one subject."""
    img = nib.load(nifti_path)
    data = img.get_fdata()  # (X, Y, Z, T)
    n_slices, n_timepoints = data.shape[2], data.shape[3]

    pairs = []
    for z in range(n_slices):
        for t in range(1, n_timepoints):
            pairs.append({
                "subject": subject_id,
                "slice": z,
                "t": t,
                "moving": data[:, :, z, t],
                "fixed": data[:, :, z, t - 1],
            })
    return pairs


from bids_discover_func import find_func_scans

def build_dataset_pairs(bids_root: str, task: str = None):
    scans = find_func_scans(bids_root, task=task)
    scans = scans[:20]  # limit to first 20 scans for testing
    all_pairs = []
    i=0
    for scan in scans:
        i+=1
        if i>20:
            break
        # use subject+session as the grouping key so pairs never cross sessions
        print(f"Building pairs for {scan.subject} | ses-{scan.session or '(no session)'} | ")
        subject_key = f"{scan.subject}_{scan.session}" if scan.session else scan.subject
        all_pairs.extend(build_pairs_for_subject(scan.path, subject_key))
    subject_ids = sorted({f"{s.subject}_{s.session}" if s.session else s.subject for s in scans})
    return all_pairs, subject_ids


def split_by_subject(pairs, subject_ids, val_fraction=VAL_FRACTION, seed=42):
    """Split by subject (not by pair) to avoid leaking adjacent timepoints across splits."""
    rng = np.random.RandomState(seed)
    subjects = list(subject_ids)
    rng.shuffle(subjects)
    n_val = max(1, int(len(subjects) * val_fraction))
    val_subjects = set(subjects[:n_val])
    train_subjects = set(subjects[n_val:])

    train_pairs = [p for p in pairs if p["subject"] in train_subjects]
    val_pairs = [p for p in pairs if p["subject"] in val_subjects]
    return train_pairs, val_pairs


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------
class SpinalCordMotionDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        p = self.pairs[idx]
        moving = pad_or_crop(normalize(p["moving"]))
        fixed = pad_or_crop(normalize(p["fixed"]))
        moving_t = torch.from_numpy(moving).unsqueeze(0).float()  # (1, H, W)
        fixed_t = torch.from_numpy(fixed).unsqueeze(0).float()
        return moving_t, fixed_t


# --------------------------------------------------------------------------
# Model / loss
# --------------------------------------------------------------------------
def build_model():
    model = vxm.torch.networks.VxmDense(
        inshape=TARGET_SHAPE,
        nb_unet_features=[[16, 32, 32, 32], [32, 32, 32, 32, 32, 16, 16]],
        int_steps=INT_STEPS,
        int_downsize=2 if INT_STEPS > 0 else 1,
    )
    return model.to(DEVICE)


def build_losses():
    similarity_loss = vxm.torch.losses.NCC(win=NCC_WINDOW).loss
    smoothness_loss = vxm.torch.losses.Grad('l2', loss_mult=2).loss
    return similarity_loss, smoothness_loss


# --------------------------------------------------------------------------
# Training / validation loops
# --------------------------------------------------------------------------
def run_epoch(model, loader, similarity_loss, smoothness_loss, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, total_sim, total_smooth, n_batches = 0.0, 0.0, 0.0, 0

    with torch.set_grad_enabled(is_train):
        for moving, fixed in loader:
            moving, fixed = moving.to(DEVICE), fixed.to(DEVICE)

            moved, flow = model(moving, fixed)
            sim = similarity_loss(fixed, moved)
            smooth = smoothness_loss(None, flow)
            loss = sim + LAMBDA_SMOOTH * smooth

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            total_sim += sim.item()
            total_smooth += smooth.item()
            n_batches += 1

    return total_loss / n_batches, total_sim / n_batches, total_smooth / n_batches


def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    print("Building pair list from NIfTI files...")
    pairs, subject_ids = build_dataset_pairs(DATA_DIR)
    print(f"Found {len(pairs)} sequential slice pairs across {len(subject_ids)} subjects.")
    train_pairs, val_pairs = split_by_subject(pairs, subject_ids)
    print(f"Total pairs: {len(pairs)} | train: {len(train_pairs)} | val: {len(val_pairs)}")
    print(f"Train subjects: {len(subject_ids) - max(1, int(len(subject_ids) * VAL_FRACTION))} "
          f"| Val subjects: {max(1, int(len(subject_ids) * VAL_FRACTION))}")

    train_ds = SpinalCordMotionDataset(train_pairs)
    val_ds = SpinalCordMotionDataset(val_pairs)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=4, pin_memory=True)

    model = build_model()
    similarity_loss, smoothness_loss = build_losses()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float("inf")

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_sim, train_smooth = run_epoch(
            model, train_loader, similarity_loss, smoothness_loss, optimizer
        )
        val_loss, val_sim, val_smooth = run_epoch(
            model, val_loader, similarity_loss, smoothness_loss, optimizer=None
        )

        print(f"Epoch {epoch:03d}/{NUM_EPOCHS} | "
              f"train loss {train_loss:.4f} (sim {train_sim:.4f}, smooth {train_smooth:.4f}) | "
              f"val loss {val_loss:.4f} (sim {val_sim:.4f}, smooth {val_smooth:.4f})")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "best_model.pt"))
            print(f"  -> saved new best model (val loss {val_loss:.4f})")

    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "final_model.pt"))
    print("Training complete.")


if __name__ == "__main__":
    main()
