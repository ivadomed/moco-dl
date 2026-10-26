"""
Slicewise rigid (rotation + translation) motion correction for spinal cord fMRI.

Pipeline:
- Discover BIDS func BOLD scans (robust to optional ses- folders)
- Reference strategy: average volume across all timepoints, per subject/slice
  (chosen to avoid the drift seen with sequential t-vs-t-1 registration, since
  every frame registers independently against the same stable target)
- Build pairs per slice: (moving=slice_t, fixed=mean_slice_across_T) for every t
- Training pairs draw from a per-subject synthetic-motion realization: a single
  temporally-correlated random-walk translation trajectory across timepoints,
  applied identically to every slice at a given timepoint (matching real motion,
  which displaces a whole acquired volume at once, not each slice independently).
  The mean/"fixed" reference is computed from this *shifted* volume, so it
  reflects realistic motion blur rather than being an unrealistically clean
  average of unperturbed frames. Nearest-neighbor interpolation only.
- Train a rigid registration network (rotation + translation only, no dense
  deformation field) with LNCC similarity loss
- No segmentation used (raw images only, per current project decision)

Requires: pip install monai nibabel torch
"""

import os

import glob
from functools import lru_cache
import random
import numpy as np
from scipy.ndimage import shift as ndi_shift
import nibabel as nib
import torch
from torch.utils.data import Sampler
from monai.data import Dataset as MonaiDataset, DataLoader as MonaiDataLoader
from monai.losses import LocalNormalizedCrossCorrelationLoss
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wandb

from rigid_model import RigidRegistrationNet

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
DATA_DIR = "crop_dataset"   # common folder containing training/ and validation/ subfolders,
                    # each holding one 4D NIfTI per subject directly (non-BIDS)
TARGET_SHAPE = (64, 64)                # fixed in-plane size after crop/pad -- sized for
                                        # the ~50x50 spinal-cord crop, with margin for
                                        # subject-to-subject variation. Was (128, 128)
                                        # before cropping; keep this matched to your actual
                                        # crop size, not the old full-FOV size.
BATCH_SIZE = 16
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4
NCC_WINDOW = 5                         # LNCC kernel size, scaled down from 9 (at 128x128)
                                        # proportionally to the new TARGET_SHAPE
MAX_SYNTHETIC_SHIFT_PX = 20.0           # max |synthetic translation| in pixels along each
                                        # axis, applied to training data only -- tune
                                        # relative to TARGET_SHAPE and expected real motion
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = "claude/fmri_moco/checkpoints_data_augmented_20260924"  # where to save best/final model checkpoints

WANDB_PROJECT = "spinal-cord-fmri-moco"
WANDB_RUN_NAME = None                  # None -> wandb auto-generates a name
N_VIZ_SAMPLES = 4                      # number of validation pairs to visualize each epoch
VOLUME_CACHE_SIZE = 4                  # max number of full 4D volumes kept decompressed in memory per worker


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
# Pair construction (index only -- no pixel data loaded here)
# --------------------------------------------------------------------------
def build_pairs_for_subject(nifti_path: str, subject_id: str):
    """
    Build pair *references* for every slice/timepoint of one subject: each
    entry is registered against that slice's average-across-T reference, so
    every timepoint (including t=0) is included -- there's no t-1 dependency
    to exclude the first frame for, unlike the previous sequential approach.
    Only reads the NIfTI header (img.shape) -- no pixel data is loaded or stored.
    """
    img = nib.load(nifti_path)   # header-only, cheap
    n_slices, n_timepoints = img.shape[2], img.shape[3]

    pairs = []
    for z in range(n_slices):
        for t in range(n_timepoints):
            pairs.append({
                "subject": subject_id,
                "path": nifti_path,
                "slice": z,
                "t": t,
            })
    return pairs


def discover_volumes(folder: str):
    """Find every NIfTI file directly under `folder` (non-recursive, non-BIDS)."""
    paths = sorted(
        glob.glob(os.path.join(folder, "*.nii.gz")) +
        glob.glob(os.path.join(folder, "*.nii"))
    )
    if not paths:
        raise FileNotFoundError(f"No .nii/.nii.gz files found in {folder}")
    return paths


def build_pairs_for_split(folder: str):
    """
    Build the full pair index for one split (training/ or validation/): every
    NIfTI file directly under `folder` is treated as one subject.
    """
    paths = discover_volumes(folder)
    all_pairs, subject_ids = [], []
    for path in paths:
        subject_id = os.path.basename(path).replace(".nii.gz", "").replace(".nii", "")
        subject_ids.append(subject_id)
        print(f"Building pairs for {subject_id}")
        all_pairs.extend(build_pairs_for_subject(path, subject_id))
    return all_pairs, subject_ids


# --------------------------------------------------------------------------
# Lazy volume loading (bounded LRU cache, per worker process)
# --------------------------------------------------------------------------
# Slicing directly into a gzip-compressed NIfTI's dataobj proxy is extremely slow
# for random access (gzip doesn't support efficient seeks -- each slice re-decompresses
# from the start). Instead we decompress a full volume once and cache a bounded number
# of them in memory: fast slice access afterward, and memory stays capped at
# VOLUME_CACHE_SIZE volumes per worker instead of the whole dataset's worth of slices.
@lru_cache(maxsize=VOLUME_CACHE_SIZE)
def _load_volume(path: str) -> np.ndarray:
    return nib.load(path).get_fdata(dtype=np.float32)  # (X, Y, Z, T)


@lru_cache(maxsize=VOLUME_CACHE_SIZE)
def _load_mean_volume(path: str) -> np.ndarray:
    """Average across the time axis, computed once per volume and cached alongside it."""
    return _load_volume(path).mean(axis=-1)  # (X, Y, Z)


def load_pair(entry: dict):
    """MONAI Dataset transform: turns one index entry into a (moving, fixed) tensor pair.
    Unaugmented -- used for validation, so val loss reflects real-data performance."""
    volume = _load_volume(entry["path"])
    mean_volume = _load_mean_volume(entry["path"])
    z, t = entry["slice"], entry["t"]

    moving = pad_or_crop(normalize(volume[:, :, z, t]))
    fixed = pad_or_crop(normalize(mean_volume[:, :, z]))

    moving_t = torch.from_numpy(moving.copy()).unsqueeze(0).float()  # (1, H, W)
    fixed_t = torch.from_numpy(fixed.copy()).unsqueeze(0).float()
    return moving_t, fixed_t


def _generate_motion_trajectory(n_timepoints: int, max_shift: float) -> np.ndarray:
    """
    Random-walk translation trajectory across timepoints: shape (T, 2), each row
    (dy, dx). Each step adds a small random increment to the previous shift, so
    consecutive timepoints have correlated displacement -- matching how real
    subject motion drifts smoothly over time rather than jumping independently
    from one frame to the next. Clipped to stay within [-max_shift, max_shift].
    """
    step_std = max_shift   # tuned so the random walk stays roughly within bounds
    steps = np.random.normal(0.0, step_std, size=(n_timepoints, 2))
    trajectory = np.cumsum(steps, axis=0)
    return np.clip(trajectory, -max_shift, max_shift)


@lru_cache(maxsize=VOLUME_CACHE_SIZE)
def _load_augmented_volume_and_mean(path: str):
    """
    Builds one synthetic-motion realization for this subject/volume:

    - A single temporally-correlated random-walk trajectory across all T
      timepoints (see _generate_motion_trajectory).
    - The SAME (dy, dx) shift at timepoint t is applied to every slice z of
      that timepoint -- matching how real patient motion at a given instant
      displaces the whole volume being acquired around that time, not each
      2D slice independently.
    - The mean/"fixed" reference is computed from this *shifted* volume, so
      it reflects the same motion blurring a real averaged image would have,
      instead of being an unrealistically clean average of unperturbed frames.

    Cached per path (bounded by VOLUME_CACHE_SIZE, per worker), same pattern as
    _load_volume/_load_mean_volume -- this also means each cache eviction (which
    happens roughly once per epoch per subject in practice, since the cache is
    far smaller than the number of subjects) regenerates a fresh random
    trajectory, giving augmentation diversity across epochs "for free" without
    needing an explicit epoch-keyed cache.
    """
    volume = _load_volume(path)  # (X, Y, Z, T) -- raw, unperturbed
    X, Y, Z, T = volume.shape

    trajectory = _generate_motion_trajectory(T, MAX_SYNTHETIC_SHIFT_PX)  # (T, 2)

    shifted = np.empty_like(volume)
    for t in range(T):
        dy, dx = trajectory[t]
        # shift the whole (X, Y, Z) block at once -- same in-plane translation
        # for every slice at this timepoint, no shift along the Z axis
        shifted[:, :, :, t] = ndi_shift(volume[:, :, :, t], shift=(dy, dx, 0),
                                         order=0, mode="constant", cval=0.0)

    mean_shifted = shifted.mean(axis=-1)  # (X, Y, Z) -- reflects motion blur
    return shifted, mean_shifted


def load_pair_train(entry: dict):
    """
    Training transform: draws from the per-subject synthetic-motion realization
    above, so that (a) the moving frame's misalignment is temporally coherent
    across timepoints rather than i.i.d. per pair, and (b) the "fixed" mean
    reference reflects that same synthetic motion rather than being computed
    from clean, unperturbed data.
    """
    shifted_volume, mean_shifted = _load_augmented_volume_and_mean(entry["path"])
    z, t = entry["slice"], entry["t"]

    moving = pad_or_crop(normalize(shifted_volume[:, :, z, t]))
    fixed = pad_or_crop(normalize(mean_shifted[:, :, z]))

    moving_t = torch.from_numpy(moving.copy()).unsqueeze(0).float()  # (1, H, W)
    fixed_t = torch.from_numpy(fixed.copy()).unsqueeze(0).float()
    return moving_t, fixed_t




class GroupedShuffleSampler(Sampler):
    """
    Shuffles pair indices in a way that stays friendly to the bounded LRU volume
    cache. A plain shuffle=True randomizes globally across every subject, which
    with only VOLUME_CACHE_SIZE volumes cached per worker causes near-constant
    cache evictions (each a multi-second full-volume decompress) once there are
    more subjects than the cache size -- this is what was making epochs effectively
    never finish.

    Instead: group indices by source file, shuffle the *order of groups* each
    epoch, and shuffle *within* each group. Indices from the same file stay
    contiguous, so a worker only needs to keep that one volume in cache for as
    long as it's working through that block -- while still fully shuffling both
    which subject comes next and the slice/timepoint order within it.
    """
    def __init__(self, pairs, seed=0):
        self.seed = seed
        self.epoch = 0
        groups = {}
        for idx, p in enumerate(pairs):
            groups.setdefault(p["path"], []).append(idx)
        self.groups = list(groups.values())
        self._length = len(pairs)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        groups = [g[:] for g in self.groups]
        for g in groups:
            rng.shuffle(g)
        rng.shuffle(groups)
        for g in groups:
            yield from g

    def __len__(self):
        return self._length


# --------------------------------------------------------------------------
# Model / loss
# --------------------------------------------------------------------------
def build_model():
    return RigidRegistrationNet().to(DEVICE)


import torch
import torch.nn as nn
# Assuming LocalNormalizedCrossCorrelationLoss is already imported from MONAI or your codebase


class GlobalNormalizedCrossCorrelationLoss(nn.Module):
    """
    Computes the Global Normalized Cross-Correlation (GNCC) over the entire image.
    Perfect for rigid registration as it provides a massive gradient catchment basin.
    """
    def __init__(self, epsilon: float = 1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Flatten spatial dimensions (keeps Batch and Channel dims intact)
        b, c = pred.shape[0], pred.shape[1]
        pred_flat = pred.view(b, c, -1)
        target_flat = target.view(b, c, -1)

        # 1. Compute global means
        pred_mean = pred_flat.mean(dim=-1, keepdim=True)
        target_mean = target_flat.mean(dim=-1, keepdim=True)

        # 2. Mean-center the tensors
        pred_centered = pred_flat - pred_mean
        target_centered = target_flat - target_mean

        # 3. Compute covariance (numerator) and variances (denominator)
        covar = (pred_centered * target_centered).sum(dim=-1)
        pred_var = (pred_centered ** 2).sum(dim=-1)
        target_var = (target_centered ** 2).sum(dim=-1)

        # 4. Calculate correlation coefficient
        denominator = torch.sqrt(pred_var * target_var) + self.epsilon
        ncc = covar / denominator

        # 5. Return loss (1 - NCC to minimize)
        return 1.0 - ncc.mean()



class CombinedNCCLoss(nn.Module):
    """
    Combines Global NCC for large catchment basins and Local NCC for fine-grained, 
    bias-field-robust sub-pixel alignment.
    """
    def __init__(self, window_size=5, global_weight=1.0, local_weight=1.0):
        super().__init__()
        self.global_ncc = GlobalNormalizedCrossCorrelationLoss()
        self.local_ncc = LocalNormalizedCrossCorrelationLoss(spatial_dims=2, kernel_size=window_size)
        
        self.global_weight = global_weight
        self.local_weight = local_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss_global = self.global_ncc(pred, target)
        loss_local = self.local_ncc(pred, target)
        
        return (self.global_weight * loss_global) + (self.local_weight * loss_local)

def build_losses():
    # Use a 1:1 ratio by default, though you can tweak these weights 
    # if one dominates the gradients early in training.
    similarity_loss = CombinedNCCLoss(window_size=NCC_WINDOW, global_weight=1.0, local_weight=1.0)
    return similarity_loss


# --------------------------------------------------------------------------
# wandb visualization
# --------------------------------------------------------------------------
def make_registration_figure(moving, fixed, warped, n_samples=N_VIZ_SAMPLES):
    """
    Build a matplotlib figure with one row per sample and columns:
    moving | fixed | warped | |warped - fixed| (absolute diff)
    All inputs are (B, 1, H, W) tensors already on CPU.
    """
    n_samples = min(n_samples, moving.shape[0])
    fig, axes = plt.subplots(n_samples, 4, figsize=(12, 3 * n_samples))
    if n_samples == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["Moving (t)", "Fixed (mean)", "Warped", "|Warped - Fixed|"]
    for row in range(n_samples):
        mov = moving[row, 0].numpy()
        fix = fixed[row, 0].numpy()
        warp = warped[row, 0].numpy()
        diff = np.abs(warp - fix)

        for col, (img, cmap) in enumerate(
            [(mov, "gray"), (fix, "gray"), (warp, "gray"), (diff, "magma")]
        ):
            ax = axes[row, col]
            ax.imshow(img, cmap=cmap)
            ax.axis("off")
            if row == 0:
                ax.set_title(col_titles[col], fontsize=10)

    fig.tight_layout()
    return fig


def log_validation_images(model, viz_moving, viz_fixed, epoch):
    """Run the model on a fixed validation batch and log a comparison figure to wandb."""
    model.eval()
    with torch.no_grad():
        viz_moving_dev = viz_moving.to(DEVICE)
        viz_fixed_dev = viz_fixed.to(DEVICE)
        warped, _ = model(viz_moving_dev, viz_fixed_dev)

    fig = make_registration_figure(viz_moving.cpu(), viz_fixed.cpu(), warped.cpu())
    wandb.log({"val/registration_examples": wandb.Image(fig)}, step=epoch)
    plt.close(fig)


# --------------------------------------------------------------------------
# Training / validation loops
# --------------------------------------------------------------------------
def run_epoch(model, loader, similarity_loss, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, n_batches = 0.0, 0

    with torch.set_grad_enabled(is_train):
        for moving, fixed in loader:
            moving, fixed = moving.to(DEVICE), fixed.to(DEVICE)

            warped, _ = model(moving, fixed)
            loss = similarity_loss(warped, fixed)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            n_batches += 1

    return total_loss / n_batches


def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_RUN_NAME,
        config={
            "target_shape": TARGET_SHAPE,
            "batch_size": BATCH_SIZE,
            "num_epochs": NUM_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "ncc_window": NCC_WINDOW,
            "architecture": "rigid (rotation + translation)",
            "reference": "average across timepoints",
        },
    )

    print("Building pair list from training/ and validation/ folders...")
    train_pairs, train_subjects = build_pairs_for_split(os.path.join(DATA_DIR, "train"))
    val_pairs, val_subjects = build_pairs_for_split(os.path.join(DATA_DIR, "validation"))
    print(f"Train: {len(train_pairs)} pairs across {len(train_subjects)} subjects "
          f"| Val: {len(val_pairs)} pairs across {len(val_subjects)} subjects")

    train_ds = MonaiDataset(data=train_pairs, transform=load_pair_train)
    val_ds = MonaiDataset(data=val_pairs, transform=load_pair)
    train_sampler = GroupedShuffleSampler(train_pairs)
    train_loader = MonaiDataLoader(train_ds, batch_size=BATCH_SIZE, sampler=train_sampler,
                                    num_workers=4, pin_memory=True)
    val_loader = MonaiDataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                                  num_workers=4, pin_memory=True)

    # fixed batch used every epoch so visualizations are directly comparable over training
    viz_moving, viz_fixed = next(iter(val_loader))

    model = build_model()
    similarity_loss = build_losses()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float("inf")

    for epoch in range(1, NUM_EPOCHS + 1):
        train_sampler.set_epoch(epoch)
        train_loss = run_epoch(model, train_loader, similarity_loss, optimizer)
        val_loss = run_epoch(model, val_loader, similarity_loss, optimizer=None)

        print(f"Epoch {epoch:03d}/{NUM_EPOCHS} | train loss {train_loss:.4f} | val loss {val_loss:.4f}")

        wandb.log({
            "epoch": epoch,
            "train/loss": train_loss,
            "val/loss": val_loss,
        }, step=epoch)

        log_validation_images(model, viz_moving, viz_fixed, epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "best_model.pt"))
            print(f"  -> saved new best model (val loss {val_loss:.4f})")

    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "final_model.pt"))
    print("Training complete.")
    wandb.finish()


if __name__ == "__main__":
    main()