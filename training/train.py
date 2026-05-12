#!/usr/bin/env python3
"""
moco-dl training: on-the-fly augmentation + DenseNet + temporal head + composite loss.

This replaces the original moco_main.py while keeping the same CLI, so it
slots into the existing workflow:

    python train.py /path/to/project_base /path/to/prepared_dataset <run_name1> [<run_name2>]

Key differences from the original
---------------------------------
- Augmentation is applied per __getitem__ from the CLEAN volume, so every
  epoch sees fresh motion (no more pre-baked aug_*.nii.gz files).
- Ground-truth (Tx, Ty, Theta) from the augmenter is part of the batch and
  drives a supervised regression loss term.
- Model is the DenseNet backbone + 1D temporal Conv head from model.py.
- Loss is the masked LNCC + supervised regression + smoothness in loss.py.

Dataset format
--------------
Each .pt file is expected to contain a dict with at least:
    "clean"  : (1, H, W, D, T)  — clean source volume (preferred)
    "moving" : (1, H, W, D, T)  — fallback if "clean" is absent (treated as clean,
                                  with a one-shot warning)
    "fixed"  : (1, H, W, D) or (1, H, W, D, T)
    "mask"   : (1, H, W, D)
    "affine" : (4, 4)
The dataset.json file (training/validation/testing entries with "data" paths)
follows the same convention as the original.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint)
from torch import optim
from torch.utils.data import DataLoader, Dataset

from model import DenseRigidNet, RigidSliceWarp
from loss import MocoLoss
from motion_augmentation import MotionAugmenter


# Suppress FutureWarning from torch.load(weights_only=...) noise; we know.
warnings.filterwarnings("ignore", category=FutureWarning)


def _fig_to_array(fig):
    """Render a matplotlib Figure into an (H, W, 3) uint8 numpy array."""
    import numpy as np
    fig.canvas.draw()
    # buffer_rgba is the modern matplotlib API; works on all backends
    buf = np.asarray(fig.canvas.buffer_rgba())
    return buf[..., :3].copy()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MocoDataset(Dataset):
    """
    Loads a clean .pt sample and applies on-the-fly motion augmentation.

    Returns a dict with:
        moving : (1, H, W, D, T) float32 — motion-augmented volume
        fixed  : (1, H, W, D) or (1, H, W, D, T)
        mask   : (1, H, W, D)
        Tx_gt, Ty_gt, Theta_gt : (D, T) — applied translations / rotation
        affine : (4, 4)
        applied_motion : bool
        data_path : str
    """

    _missing_clean_warned = False

    def __init__(self, file_list: list[dict], data_dir: Path,
                 augmenter: MotionAugmenter | None,
                 augment: bool = True,
                 t_window: int | None = None):
        """
        t_window : if set, randomly sample a contiguous window of `t_window`
                   timepoints per sample. This is the primary memory-control
                   knob for long fMRI runs (T=300+). A typical setting is
                   64 or 96 — gives ~5x memory reduction at T=300.
                   Set to None or 0 (or to a value >= T) to use the full series.
        """
        self.file_list = file_list
        self.data_dir = Path(data_dir)
        self.augmenter = augmenter
        self.augment = augment
        self.t_window = t_window if t_window and t_window > 0 else None

    def __len__(self) -> int:
        return len(self.file_list)

    def _load(self, idx: int) -> dict[str, Any]:
        sample = self.file_list[idx]
        pt_path = sample["data"]
        final_path = (Path(pt_path) if os.path.isabs(pt_path)
                      else self.data_dir / pt_path)
        return torch.load(final_path, weights_only=False), str(final_path)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        data, path = self._load(idx)

        # Pick the clean volume; fall back to "moving" with a one-shot warning
        if "clean" in data:
            clean = data["clean"]
        elif "moving" in data:
            if not MocoDataset._missing_clean_warned:
                print("WARNING: dataset uses legacy 'moving' field; treating it as "
                      "clean. Re-run dataset_preparation.py to save 'clean' explicitly.",
                      file=sys.stderr)
                MocoDataset._missing_clean_warned = True
            clean = data["moving"]
        else:
            raise KeyError(f"{path}: no 'clean' or 'moving' key in sample")

        fixed = data["fixed"]
        mask = data["mask"]
        affine = data.get("affine", torch.eye(4))

        # ---- Temporal subsampling (memory control for long fMRI) ----
        # Take a contiguous random window of `t_window` timepoints, so the
        # network sees a different chunk of each subject every epoch.
        # Slice the clean volume AND any per-timepoint fixed reference.
        T_full = clean.shape[-1]
        if self.t_window is not None and self.t_window < T_full:
            # contiguous window — preserves temporal smoothness within the window
            start = int(np.random.randint(0, T_full - self.t_window + 1))
            stop = start + self.t_window
            clean = clean[..., start:stop]
            # If fixed is 4D (dMRI: per-timepoint reference) slice it too;
            # otherwise it's 3D (fMRI: temporal mean), nothing to slice.
            if fixed.ndim == 5:                # (1, H, W, D, T)
                fixed = fixed[..., start:stop]

        # Tensors on disk are already (C, H, W, D, [T]) with C=1.
        # DataLoader will add the batch axis. So the model sees:
        #   moving: (B, 1, H, W, D, T)
        #   fixed:  (B, 1, H, W, D) for fMRI or (B, 1, H, W, D, T) for dMRI
        #   mask:   (B, 1, H, W, D)
        # No squeeze/unsqueeze needed.

        # Augmentation runs on numpy in (H, W, D, T) layout
        if self.augment and self.augmenter is not None:
            # clean is (1, H, W, D, T) — drop channel axis for numpy work
            clean_np = clean[0].cpu().numpy().astype(np.float32)
            mask_np = mask[0].cpu().numpy().astype(np.float32)
            out = self.augmenter(clean_np, mask_np)
            moving = torch.from_numpy(out["moving"]).unsqueeze(0)  # back to (1,H,W,D,T)
            Tx_gt = torch.from_numpy(out["Tx_gt"])
            Ty_gt = torch.from_numpy(out["Ty_gt"])
            applied_motion = out["applied_motion"]
        else:
            moving = clean.clone()
            D, T = clean.shape[-2], clean.shape[-1]
            Tx_gt = torch.zeros(D, T)
            Ty_gt = torch.zeros(D, T)
            applied_motion = False

        # Theta GT is not exposed by the augmenter — pass zeros + a flag.
        Theta_gt = torch.zeros_like(Tx_gt)

        return {
            "moving": moving.float(),
            "fixed": fixed.float(),
            "mask": mask.float(),
            "Tx_gt": Tx_gt.float(),
            "Ty_gt": Ty_gt.float(),
            "Theta_gt": Theta_gt.float(),
            "has_theta_gt": False,
            "affine": affine,
            "applied_motion": applied_motion,
            "data_path": path,
        }


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------

class MocoDataModule(pl.LightningDataModule):
    def __init__(self, json_path: Path, data_dir: Path,
                 batch_size: int, num_workers: int,
                 augmenter_train: MotionAugmenter,
                 augmenter_val: MotionAugmenter | None = None,
                 t_window_train: int | None = None,
                 t_window_val: int | None = None):
        super().__init__()
        self.json_path = json_path
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.augmenter_train = augmenter_train
        self.augmenter_val = augmenter_val if augmenter_val is not None else augmenter_train
        self.t_window_train = t_window_train
        self.t_window_val = t_window_val

    def setup(self, stage: str | None = None):
        with open(self.json_path) as f:
            ds = json.load(f)
        self.train_ds = MocoDataset(ds["training"], self.data_dir,
                                    augmenter=self.augmenter_train, augment=True,
                                    t_window=self.t_window_train)
        self.val_ds = MocoDataset(ds["validation"], self.data_dir,
                                  augmenter=self.augmenter_val, augment=True,
                                  t_window=self.t_window_val)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers,
                          pin_memory=True, persistent_workers=self.num_workers > 0)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers,
                          pin_memory=True, persistent_workers=self.num_workers > 0)


# ---------------------------------------------------------------------------
# Lightning module
# ---------------------------------------------------------------------------

class MocoTrainer(pl.LightningModule):
    """Trains DenseRigidNet against the composite loss."""

    def __init__(self,
                 lr: float = 1e-4,
                 weight_decay: float = 1e-4,
                 max_translation_vox: float = 5.0,
                 max_rotation_deg: float = 5.0,
                 w_sim: float = 1.0,
                 w_reg: float = 1.0,
                 w_smooth: float = 0.1,
                 use_checkpointing: bool = False,
                 image_log_every_n_epochs: int = 1):
        super().__init__()
        self.save_hyperparameters()
        self.model = DenseRigidNet(
            max_translation_vox=max_translation_vox,
            max_rotation_deg=max_rotation_deg,
            use_checkpointing=use_checkpointing,
        )
        self.warp = RigidSliceWarp()
        self.loss_fn = MocoLoss(w_sim=w_sim, w_reg=w_reg, w_smooth=w_smooth)
        self.lr = lr
        self.weight_decay = weight_decay
        self.image_log_every_n_epochs = image_log_every_n_epochs
        # Per-epoch caches for visual logging (cleared after each log)
        self._vis_cache: dict[str, dict] = {"train": {}, "val": {}}

    def _forward_and_warp(self, batch: dict[str, Any]):
        moving = batch["moving"]              # (B, 1, H, W, D, T)
        fixed = batch["fixed"]
        Tx, Ty, Theta = self.model(moving, fixed)   # (B, D, T) each

        # Warp slice-by-slice across t. Each per-t warp materializes
        # affine_grid + grid_sample tensors which stay in the autograd graph
        # until backward runs — for T=300 this dominates memory. Checkpoint
        # each warp call when checkpointing is enabled so its activations
        # are recomputed on the backward pass instead of stored.
        B, _, H, W, D, T = moving.shape
        warped_list = []
        ckpt = self.hparams.get("use_checkpointing", False) if hasattr(self, "hparams") else False
        for t in range(T):
            mov_t = moving[..., t]
            tx_t = Tx[..., t]
            ty_t = Ty[..., t]
            th_t = Theta[..., t]
            if ckpt and self.training and mov_t.requires_grad:
                warped_t = torch.utils.checkpoint.checkpoint(
                    self.warp, mov_t, tx_t, ty_t, th_t,
                    use_reentrant=False,
                )
            else:
                warped_t = self.warp(mov_t, tx_t, ty_t, th_t)
            warped_list.append(warped_t)
        warped = torch.stack(warped_list, dim=-1)
        return warped, Tx, Ty, Theta

    def _step(self, batch: dict[str, Any], stage: str, batch_idx: int = 0) -> torch.Tensor:
        warped, Tx, Ty, Theta = self._forward_and_warp(batch)

        # Theta_gt is currently always zero (augmenter doesn't expose it),
        # so we pass None to avoid biasing the model toward zero rotation.
        # The MocoLoss falls back to a tiny L2 penalty on Theta in that case.
        Theta_gt_or_none = None
        if bool(batch.get("has_theta_gt", torch.tensor(False)).any() if isinstance(batch.get("has_theta_gt"), torch.Tensor) else False):
            Theta_gt_or_none = batch["Theta_gt"]

        total, comps = self.loss_fn(
            warped=warped,
            fixed=batch["fixed"],
            mask=batch["mask"],
            Tx_pred=Tx, Ty_pred=Ty, Theta_pred=Theta,
            Tx_gt=batch["Tx_gt"], Ty_gt=batch["Ty_gt"],
            Theta_gt=Theta_gt_or_none,
        )

        # Logging
        bs = batch["moving"].size(0)
        for k, v in comps.items():
            self.log(f"{stage}/{k.split('/', 1)[1]}", v, prog_bar=(k == "loss/total"),
                     batch_size=bs, on_epoch=True, on_step=False)

        # Metric: mean abs translation error (very interpretable for motion correction)
        with torch.no_grad():
            mae_xy = ((Tx - batch["Tx_gt"]).abs().mean()
                      + (Ty - batch["Ty_gt"]).abs().mean()) / 2
            self.log(f"{stage}/MAE_xy_vox", mae_xy, batch_size=bs,
                     on_epoch=True, on_step=False)

            # Cache the first batch of the epoch for visual logging.
            # We only need item 0 of the batch and we move everything to CPU
            # immediately to keep GPU memory free.
            if batch_idx == 0 and self._should_log_images():
                self._vis_cache[stage] = {
                    "moving": batch["moving"][0].detach().cpu(),
                    "clean":  batch.get("clean", batch["moving"])[0].detach().cpu()
                              if "clean" in batch else None,
                    "fixed":  batch["fixed"][0].detach().cpu(),
                    "mask":   batch["mask"][0].detach().cpu(),
                    "warped": warped[0].detach().cpu(),
                    "Tx_pred": Tx[0].detach().cpu(),
                    "Ty_pred": Ty[0].detach().cpu(),
                    "Theta_pred": Theta[0].detach().cpu(),
                    "Tx_gt": batch["Tx_gt"][0].detach().cpu(),
                    "Ty_gt": batch["Ty_gt"][0].detach().cpu(),
                    "applied_motion": (batch["applied_motion"][0].item()
                                       if isinstance(batch["applied_motion"], torch.Tensor)
                                       else bool(batch["applied_motion"])),
                }

        return total

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train", batch_idx=batch_idx)

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val", batch_idx=batch_idx)

    # -----------------------------------------------------------------------
    # Visual logging
    # -----------------------------------------------------------------------

    def _should_log_images(self) -> bool:
        """Throttle image logging by epoch."""
        n = self.image_log_every_n_epochs
        if n is None or n <= 0:
            return False
        return (self.current_epoch % n) == 0

    def _make_visual(self, stage: str):
        """
        Build a side-by-side visual: moving | warped | fixed for 4 timepoints,
        plus a small Tx/Ty trace plot (predicted vs GT).

        Returns (image_array, traces_array) — both HxWx3 uint8.
        Returns (None, None) if no cache for this stage.
        """
        cache = self._vis_cache.get(stage, {})
        if not cache:
            return None, None
        # Lazy imports — matplotlib at the top of the file would slow startup
        import matplotlib
        matplotlib.use("Agg")  # headless
        import matplotlib.pyplot as plt
        import numpy as np

        moving = cache["moving"].squeeze(0).numpy()   # (H, W, D, T)
        warped = cache["warped"].squeeze(0).numpy()
        fixed_t = cache["fixed"]
        if fixed_t.ndim == 4:
            fixed = fixed_t.squeeze(0).numpy()        # (H, W, D)
            fixed_per_t = False
        else:
            fixed = fixed_t.squeeze(0).numpy()        # (H, W, D, T)
            fixed_per_t = True

        H, W, D, T = moving.shape
        z = D // 2                                    # mid-cord slice
        # Pick 4 evenly-spaced timepoints
        t_indices = [int(round(x)) for x in np.linspace(0, T - 1, 4)]

        fig, axes = plt.subplots(4, 3, figsize=(7.5, 9))
        vmin, vmax = np.percentile(fixed, (1, 99))
        for row, t in enumerate(t_indices):
            for col, (name, img) in enumerate([
                ("moving", moving[:, :, z, t]),
                ("warped", warped[:, :, z, t]),
                ("fixed", fixed[:, :, z, t] if fixed_per_t else fixed[:, :, z]),
            ]):
                ax = axes[row, col]
                ax.imshow(img.T, cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
                ax.set_xticks([]); ax.set_yticks([])
                if row == 0:
                    ax.set_title(name, fontsize=10)
                if col == 0:
                    ax.set_ylabel(f"t={t}", fontsize=9)
        applied = cache.get("applied_motion", True)
        fig.suptitle(f"{stage}  z={z}  motion_applied={applied}", fontsize=11)
        fig.tight_layout()
        img_arr = _fig_to_array(fig)
        plt.close(fig)

        # Traces: pred vs GT, all slices overlaid, mean ± across z
        fig2, axes2 = plt.subplots(2, 1, figsize=(7.5, 4), sharex=True)
        Tx_pred = cache["Tx_pred"].numpy()           # (D, T)
        Ty_pred = cache["Ty_pred"].numpy()
        Tx_gt = cache["Tx_gt"].numpy()
        Ty_gt = cache["Ty_gt"].numpy()
        for ax, pred, gt, name in [
            (axes2[0], Tx_pred, Tx_gt, "Tx (axis-0, vox)"),
            (axes2[1], Ty_pred, Ty_gt, "Ty (axis-1, vox)"),
        ]:
            ax.plot(pred.mean(axis=0), color="C0", label="pred (mean over z)", lw=1.5)
            ax.fill_between(np.arange(pred.shape[1]),
                            pred.mean(axis=0) - pred.std(axis=0),
                            pred.mean(axis=0) + pred.std(axis=0),
                            color="C0", alpha=0.2)
            ax.plot(gt.mean(axis=0), color="C1", label="gt (mean over z)", lw=1.5, ls="--")
            ax.fill_between(np.arange(gt.shape[1]),
                            gt.mean(axis=0) - gt.std(axis=0),
                            gt.mean(axis=0) + gt.std(axis=0),
                            color="C1", alpha=0.2)
            ax.set_ylabel(name)
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(alpha=0.3)
        axes2[-1].set_xlabel("timepoint")
        fig2.suptitle(f"{stage}  predicted vs GT translations", fontsize=11)
        fig2.tight_layout()
        traces_arr = _fig_to_array(fig2)
        plt.close(fig2)

        # Free cache for this stage so we don't double-log between epochs
        self._vis_cache[stage] = {}
        return img_arr, traces_arr

    def _log_visual_check(self, stage: str):
        if not self._should_log_images():
            return
        img_arr, traces_arr = self._make_visual(stage)
        if img_arr is None:
            return
        # Only wandb supports image logging out of the box; CSV logger doesn't.
        from pytorch_lightning.loggers import WandbLogger
        if isinstance(self.logger, WandbLogger):
            import wandb
            self.logger.experiment.log({
                f"vis/{stage}/slices": wandb.Image(img_arr,
                    caption=f"{stage} epoch {self.current_epoch}"),
                f"vis/{stage}/traces": wandb.Image(traces_arr,
                    caption=f"{stage} epoch {self.current_epoch} Tx/Ty traces"),
            }, step=self.global_step)

    def on_train_epoch_end(self):
        self._log_visual_check("train")

    def on_validation_epoch_end(self):
        self._log_visual_check("val")

    def configure_optimizers(self):
        opt = optim.AdamW(self.parameters(), lr=self.lr,
                          weight_decay=self.weight_decay)
        # Cosine schedule with linear warmup — works well for this kind of
        # regression and avoids the LR-plateau patience-tuning headache.
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200, eta_min=self.lr * 0.01)
        return {"optimizer": opt,
                "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train moco-dl with on-the-fly augmentation + temporal head.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("base_path", type=Path,
                   help="Base directory; checkpoints go to base_path/trained_weights/")
    p.add_argument("data_path", type=Path,
                   help="Prepared dataset directory containing dataset.json")
    p.add_argument("run_name", type=str,
                   help="Identifier for this run; checkpoint -> <run_name>.ckpt")
    p.add_argument("finetune_run_name", type=str, nargs="?", default=None,
                   help="If set, load <run_name>.ckpt as init and save as <finetune_run_name>.ckpt")

    # Training hyperparams
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=1,
                   help="Batch size. NOTE: with mixed-shape datasets (different "
                        "subjects having different D, H, or T) you must use "
                        "batch-size=1 because the default collate can't stack "
                        "different-shaped tensors. Use --accumulate-grad-batches "
                        "to get a larger effective batch size instead.")
    p.add_argument("--accumulate-grad-batches", type=int, default=1,
                   help="Accumulate gradients over N forward passes before "
                        "stepping. Effective batch = batch-size * N, "
                        "without the same-shape constraint.")
    p.add_argument("--t-window", type=int, default=None,
                   help="If set, train on a random contiguous window of this "
                        "many timepoints per sample (e.g. 64 or 96). Primary "
                        "memory-control knob for long fMRI runs (T=300+). "
                        "Validation uses the full T by default; override with "
                        "--t-window-val.")
    p.add_argument("--t-window-val", type=int, default=None,
                   help="t-window for validation (default: same as --t-window).")
    p.add_argument("--use-checkpointing", action="store_true",
                   help="Enable gradient checkpointing on the backbone and the "
                        "per-timepoint warp. ~30% slower, but cuts activation "
                        "memory by roughly 4-5x. Use when T is large.")
    p.add_argument("--image-log-every-n-epochs", type=int, default=1,
                   help="Log a visual sanity-check (moving/warped/fixed slices + "
                        "Tx/Ty traces) every N epochs to W&B. Two visuals are "
                        "logged per cycle (one from train, one from val). "
                        "Set to 0 to disable. Default: 1 (every epoch).")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--gradient-clip", type=float, default=1.0)

    # Loss weights
    p.add_argument("--w-sim", type=float, default=1.0)
    p.add_argument("--w-reg", type=float, default=1.0)
    p.add_argument("--w-smooth", type=float, default=0.1)

    # Model bounds
    p.add_argument("--max-translation-vox", type=float, default=5.0)
    p.add_argument("--max-rotation-deg", type=float, default=5.0)

    # Augmentation overrides
    p.add_argument("--aug-p-case", type=float, default=0.85)
    p.add_argument("--aug-max-translation", type=float, default=4.0)
    p.add_argument("--aug-max-rotation", type=float, default=3.0)
    p.add_argument("--aug-max-through-plane", type=float, default=1.0)

    # Hardware
    p.add_argument("--cuda-device", type=str, default=None,
                   help="Set CUDA_VISIBLE_DEVICES (e.g. '0' or '2'). Leave unset for default.")
    p.add_argument("--precision", type=str, default="16-mixed",
                   choices=["32", "bf16-mixed", "16-mixed"])

    # Logging
    p.add_argument("--wandb-project", type=str, default="moco-dmri")
    p.add_argument("--no-wandb", action="store_true",
                   help="Disable Weights & Biases logging (use CSV logger instead).")

    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.cuda_device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device

    # Defer PyTorch CUDA-touching imports until after env vars set
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True

    pl.seed_everything(args.seed, workers=True)

    json_path = args.data_path / "dataset.json"
    if not json_path.exists():
        sys.exit(f"ERROR: {json_path} not found")

    print(f"Base path  : {args.base_path}")
    print(f"Data path  : {args.data_path}")
    print(f"JSON path  : {json_path}")
    print(f"Run name   : {args.run_name}"
          + (f"  (fine-tune from previous run, save as {args.finetune_run_name})"
             if args.finetune_run_name else ""))

    # Augmenter
    augmenter = MotionAugmenter(
        p_case=args.aug_p_case,
        max_translation_vox=args.aug_max_translation,
        max_rotation_deg=args.aug_max_rotation,
        max_through_plane_vox=args.aug_max_through_plane,
    )

    dm = MocoDataModule(
        json_path=json_path,
        data_dir=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augmenter_train=augmenter,
        t_window_train=args.t_window,
        t_window_val=args.t_window_val if args.t_window_val is not None else args.t_window,
    )

    # Model: either fresh, or loaded from a previous checkpoint as warm start
    ckpt_dir = args.base_path / "trained_weights"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if args.finetune_run_name is not None:
        prev_ckpt = ckpt_dir / f"{args.run_name}.ckpt"
        save_name = args.finetune_run_name
        if prev_ckpt.exists():
            print(f"Loading weights from {prev_ckpt}")
            model = MocoTrainer.load_from_checkpoint(
                str(prev_ckpt),
                lr=args.lr,
                weight_decay=1e-4,
                max_translation_vox=args.max_translation_vox,
                max_rotation_deg=args.max_rotation_deg,
                w_sim=args.w_sim, w_reg=args.w_reg, w_smooth=args.w_smooth,
                use_checkpointing=args.use_checkpointing,
                image_log_every_n_epochs=args.image_log_every_n_epochs,
            )
        else:
            print(f"WARNING: --finetune_run_name set but {prev_ckpt} not found; "
                  f"training from scratch under {save_name}.")
            model = MocoTrainer(
                lr=args.lr,
                max_translation_vox=args.max_translation_vox,
                max_rotation_deg=args.max_rotation_deg,
                w_sim=args.w_sim, w_reg=args.w_reg, w_smooth=args.w_smooth,
                use_checkpointing=args.use_checkpointing,
                image_log_every_n_epochs=args.image_log_every_n_epochs,
            )
    else:
        save_name = args.run_name
        model = MocoTrainer(
            lr=args.lr,
            max_translation_vox=args.max_translation_vox,
            max_rotation_deg=args.max_rotation_deg,
            w_sim=args.w_sim, w_reg=args.w_reg, w_smooth=args.w_smooth,
            use_checkpointing=args.use_checkpointing,
            image_log_every_n_epochs=args.image_log_every_n_epochs,
        )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Decide which metric to monitor for checkpointing/early-stopping.
    # If the dataset has no validation entries, fall back to train/total.
    with open(json_path) as f:
        ds = json.load(f)
    has_val = len(ds.get("validation", [])) > 0
    monitor_metric = "val/total" if has_val else "train/total"
    if not has_val:
        print("WARNING: dataset has no validation split; "
              "monitoring train/total instead of val/total.")

    # Callbacks
    ckpt_cb = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename=save_name,
        monitor=monitor_metric,
        mode="min",
        save_top_k=1,
        save_weights_only=False,
    )
    early = EarlyStopping(monitor=monitor_metric, mode="min",
                          patience=args.patience, verbose=True)
    lr_mon = LearningRateMonitor(logging_interval="epoch")

    # Logger
    if args.no_wandb:
        from pytorch_lightning.loggers import CSVLogger
        logger = CSVLogger(save_dir=str(args.base_path / "logs"),
                           name=save_name)
    else:
        from pytorch_lightning.loggers import WandbLogger
        logger = WandbLogger(project=args.wandb_project, name=save_name)
        try:
            logger.experiment.config.update(vars(args), allow_val_change=True)
        except Exception:
            pass

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        logger=logger,
        callbacks=[ckpt_cb, early, lr_mon],
        accelerator="auto",
        devices=1,
        precision=args.precision,
        gradient_clip_val=args.gradient_clip,
        accumulate_grad_batches=args.accumulate_grad_batches,
        log_every_n_steps=25,
        deterministic=False,
    )

    print(f"Start: {time.ctime()}")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    trainer.fit(model, datamodule=dm)

    print(f"Best checkpoint : {ckpt_cb.best_model_path}")
    if ckpt_cb.best_model_score is not None:
        print(f"Best val/total  : {ckpt_cb.best_model_score.item():.4f}")
    print(f"End: {time.ctime()}")


if __name__ == "__main__":
    main()
