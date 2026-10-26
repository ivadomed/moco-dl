#!/usr/bin/env python3
"""
Created on Mon Sep 1 2025
@author: Nontharat Tucksinapinunchai

DenseRigidNet: DenseNet-based Rigid Motion Correction for dMRI/fMRI

This script trains a deep learning model for motion correction in diffusion MRI (dMRI)
or functional MRI (fMRI). It uses a DenseNet-based slice-wise regressor to estimate
in-plane translations (Tx, Ty) for each slice and timepoint, guided by a mask and a
fixed reference volume.

The model is trained with a composite loss function combining:
- Global normalized cross-correlation (GNCC)
- L2 (MSE) and L1 losses inside a mask
- Regularization on translation smoothness across slices and time

Key features:
- Handles both 4D moving + 4D fixed (dMRI) or 4D moving + 3D fixed (fMRI) with mask inputs
- Normalization inside mask using 1%–99% percentiles
- Slice-wise translation estimation using DenseNet blocks
- Saves best weights, logs training/validation losses, and integrates with W&B

Usage:
------
In terminal/command line:

    python moco_main.py /path/to/project_base path/to/prepared_dataset <run_name1> <run_name2>(opt)

Arguments:
    /path/to/project_base           : base directory containing the script and trained_weights
    /path/to/prepared_dataset       : data directory depending on training dataset
    run_name1                            : identifier for this training run (checkpoint filename)
    run_name2 (opt)                      : if provided, fine-tune or continue from run1 but save under run2 name

"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import gc
import sys
import time
import torch
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.fastest = True

import json
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from torch import optim
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from monai.networks.blocks import Warp
from monai.data import DataLoader, Dataset

start_time = time.ctime()
print("Start at:", start_time)

# -----------------------------
# DataGenerator
# -----------------------------
class DataGenerator(Dataset):
    """
       Custom dataset class for loading paired dMRI/fMRI samples from .pt files.
    """
    def __init__(self, file_list, data_dir):
        self.file_list = file_list
        self.data_dir = data_dir

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        sample = self.file_list[idx]
        pt_path = sample["data"]

        # If absolute path in JSON
        final_path = pt_path if os.path.isabs(pt_path) else os.path.join(self.data_dir, pt_path)
        data = torch.load(final_path, weights_only=False)

        moving = data["moving"] if "moving" in data else data["clean"]
        fixed = data["fixed"]
        mask = data["mask"]
        affine = data["affine"]


        return {"moving": moving, "fixed": fixed, "mask": mask,
                "affine": affine, "data_path": pt_path}

# -----------------------------
# DataModule
# -----------------------------
class DataModule(pl.LightningDataModule):
    """
        PyTorch Lightning DataModule for motion correction training.
        Handles training/validation split from dataset.json and provides DataLoaders for both.
    """
    def __init__(self, json_path, data_dir, batch_size, num_workers):
        super().__init__()
        self.json_path = json_path
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        # Load JSON
        with open(self.json_path, "r") as f:
            dataset_dict = json.load(f)

        self.train_ds = DataGenerator(dataset_dict["training"], self.data_dir)
        self.val_ds = DataGenerator(dataset_dict["validation"], self.data_dir)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=True, persistent_workers=True, prefetch_factor=4)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True, persistent_workers=True, prefetch_factor=4)

# -----------------------------
# Some helper function
# -----------------------------
def _get_fixed_t(fixed: torch.Tensor, t: int) -> torch.Tensor:
    """
    Return fixed volume (B,1,H,W,D) at time t, supporting both 3D fixed (fMRI) and 4D fixed (dMRI).
    """
    if fixed.ndim == 5:
        return fixed.contiguous()
    if fixed.ndim == 6:
        if isinstance(t, torch.Tensor):
            t = int(t.item())
        return fixed[..., t].contiguous()
    raise ValueError(f"Unexpected fixed ndim={fixed.ndim}")

# -----------------------------
# Loss function (Similarity losses (LNCC + L2) and Rigid Smoothness)
# -----------------------------
def normalize_volume(x, mask, pmin=1, pmax=99, eps=1e-6):
    """
    Normalizes intensity inside mask to 0–1 range using percentile scaling.
    """
    m = (mask > 0).float()
    vals = x[m.bool()]
    if vals.numel() < 10:
        return x  # skip if mask empty
    lo = torch.quantile(vals, pmin/100)
    hi = torch.quantile(vals, pmax/100)
    return ((x - lo) / (hi - lo + eps)).clamp(0, 1) * m

def global_ncc_loss(a, b, mask, eps=1e-6):
    """
    Compute global normalized cross-correlation (GNCC) across timepoints inside mask.
    """
    B, C, H, W, D, T_w = a.shape
    T_f = b.shape[-1] if b.ndim == 6 else 1
    T = min(T_w, T_f)

    vals = []
    for t in range(T):
        a_t = normalize_volume(a[..., t], mask) * mask
        b_t = normalize_volume(_get_fixed_t(b, t), mask) * mask
        a_mean = a_t.mean(dim=(2,3,4), keepdim=True)
        b_mean = b_t.mean(dim=(2,3,4), keepdim=True)
        a0, b0 = a_t - a_mean, b_t - b_mean
        num = (a0 * b0).mean(dim=(2,3,4), keepdim=True)
        den = torch.sqrt(
            (a0.pow(2).mean(dim=(2,3,4), keepdim=True) *
             b0.pow(2).mean(dim=(2,3,4), keepdim=True))
        ).clamp_min(eps)
        vals.append(-(num / den).mean())
    return torch.stack(vals).mean()

def l2_loss(warped_all, fixed, mask):
    """
    Compute mean-squared error (L2 loss) across timepoints inside mask.
    """
    B, C, H, W, D, T_w = warped_all.shape
    T_f = fixed.shape[-1] if fixed.ndim == 6 else 1
    T = min(T_w, T_f)

    losses = []
    for t in range(T):
        warped_t = normalize_volume(warped_all[..., t], mask)
        fixed_t  = normalize_volume(_get_fixed_t(fixed, t), mask)
        losses.append(F.mse_loss(warped_t * mask, fixed_t * mask))
    return torch.stack(losses).mean()

def rigid_smoothness(Tx, Ty, lam_z=1e-4, lam_t=1e-5):
    """
    Slice-wise smoothness regularization along z-axis and timepoint.
    Tx, Ty: (B,1,D,T)
    """
    dz_x = (Tx[..., 1:, :] - Tx[..., :-1, :]).abs().mean()
    dz_y = (Ty[..., 1:, :] - Ty[..., :-1, :]).abs().mean()
    dt_x = (Tx[..., :, 1:] - Tx[..., :, :-1]).abs().mean()
    dt_y = (Ty[..., :, 1:] - Ty[..., :, :-1]).abs().mean()
    return lam_z * (dz_x + dz_y) + lam_t * (dt_x + dt_y)

# -----------------------------
# Dense Block and Layer
# -----------------------------
class DenseBlock(nn.Module):
    """
    3D DenseNet block (multiple Dense Layers) with growth connections.
    """
    def __init__(self, in_channels, growth_rate, n_layers=5):
        super().__init__()
        layers, channels = [], in_channels
        for _ in range(n_layers):
            layers.append(nn.Sequential(        # Dense Layer: IN → ReLU → Conv(3x3)
                nn.InstanceNorm3d(channels, affine=True, track_running_stats=False),
                nn.ReLU(inplace=True),
                nn.Conv3d(channels, growth_rate, 3, padding=1, bias=False)
            ))
            channels += growth_rate
        self.block = nn.ModuleList(layers)

    def forward(self, x):
        for l in self.block:
            out = l(x)
            x = torch.cat([x, out], dim=1)
        return x

# -----------------------------
# DenseNet
# -----------------------------
class DenseNetRegressorSliceWise(nn.Module):
    """
    DenseNet backbone that predicts slice-wise in-plane translation (Tx, Ty) for each z-slice in the 3D volume.
    Output shape: (B, 3, D)
    """
    def __init__(self, in_channels=2, growth_rate=8, num_blocks=2, max_vox_shift=5.0):
        super().__init__()
        self.conv_in = nn.Conv3d(in_channels, 16, (3, 3, 1), stride=(2, 2, 1), padding=(1, 1, 0))
        self.blocks = nn.ModuleList()
        self.max_vox_shift = max_vox_shift
        self.shift_scale = nn.Parameter(torch.tensor(1.0))  # learnable >0 via softplus

        ch = 16
        for _ in range(num_blocks):
            blk = DenseBlock(ch, growth_rate)   # DenseBlock
            self.blocks.append(blk)
            ch += growth_rate * 5
            self.blocks.append(nn.Sequential(       # Transition Layer: IN → ReLU → Conv(1x1) → MaxPool(2x2)
                nn.InstanceNorm3d(ch, affine=True, track_running_stats=False),
                nn.ReLU(inplace=True),
                nn.Conv3d(ch, ch // 2, 1, bias=False),
                nn.MaxPool3d((2, 2, 1))     # sharper feature propagation
            ))
            ch = ch // 2

        # keep slice (depth) dimension, pool only H,W
        self.slice_pool = nn.AdaptiveAvgPool3d((1, 1, None))
        self.conv_out = nn.Conv1d(ch, 2, kernel_size=1)  # output Tx, Ty, Th per slice

        # initialize output near zero (identity)
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    def forward(self, x):
        x = self.conv_in(x)
        for b in self.blocks:
            x = b(x)
        x = x.mean(dim=(2,3))           # (B, C, D')
        theta = self.conv_out(x)        # (B, 2, D')

        s = torch.nn.functional.softplus(self.shift_scale) + 1e-4
        Ty = torch.tanh(theta[:, 0:1, :]) * (self.max_vox_shift * s)
        Tx = torch.tanh(theta[:, 1:2, :]) * (self.max_vox_shift * s)
        return torch.cat([Tx, Ty], dim=1)

# -----------------------------
# Warp Function
# -----------------------------
class RigidWarp(nn.Module):
    """
    Slice-wise rigid warp using MONAI's Warp (voxel-unit flow).
    """
    def __init__(self, mode="bilinear", padding_mode="border"):
        super().__init__()
        self.warper = Warp(mode=mode, padding_mode=padding_mode)

    def build_field(self, Tx, Ty, vol):
        """
        Args:
            Tx, Ty: (B,1,D) voxel shifts
            vol: (B,1,H,W,D)
        Returns:
            flow: (B,3,H,W,D) voxel-unit displacement field
        """
        B, C, H, W, D = vol.shape

        # Expand slice-wise translations Tx, Ty into full 3D displacement field
        # Tx_field: (B,1,H,W,D)
        Tx_field = Tx[:, :, None, None, :].expand(B, 1, H, W, D)
        Ty_field = Ty[:, :, None, None, :].expand(B, 1, H, W, D)
        Tz_field = torch.zeros_like(Tx_field)

        return torch.cat([Tx_field, Ty_field, Tz_field], dim=1)

    def forward(self, vol, Tx, Ty):
        """
        vol: (B,1,H,W,D)
        Tx,Ty: (B,1,D)
        """
        flow = self.build_field(Tx, Ty, vol)
        warped = self.warper(vol, flow)     # MONAI Warp does the 3D spatial transform
        return warped, flow

# -----------------------------
# Main Model
# -----------------------------
class DenseRigidReg(pl.LightningModule):
    """
    Main PyTorch Lightning module for DenseNet-based rigid motion correction.
    """
    def __init__(self, lr=1e-4):
        super().__init__()
        self.lr = lr
        self.backbone = DenseNetRegressorSliceWise(in_channels=2, max_vox_shift=3.0)
        self.warp = RigidWarp(mode="bilinear", padding_mode="border")

    def forward(self, moving, fixed, mask):
        """
            moving: (B,1,H,W,D,T)
            fixed:  (B,1,H,W,D) or (B,1,H,W,D,T)
            mask:   (B,1,H,W,D)
            Returns:
                warped_all: (B,1,H,W,D,T)
                Tx_all, Ty_all: (B,1,D,T)
            """
        B, _, H, W, D, T = moving.shape
        warped_list, flow_list, Tx_list, Ty_list = [], [], [], []

        for t in range(T):
            mov_t = moving[..., t]
            fix_t = _get_fixed_t(fixed, t)

            # Normalize at the same resolution as the network input
            mov_norm_ds = normalize_volume(mov_t, mask)
            fix_norm_ds = normalize_volume(fix_t, mask)

            x = torch.cat([mov_norm_ds, fix_norm_ds], dim=1)

            theta = self.backbone(x)  # (B, 2, D')
            Ty = theta[:, 0:1, :]
            Tx = theta[:, 1:2, :]

            warped, flow = self.warp(mov_t, Tx, Ty)
            warped_list.append(warped)
            flow_list.append(flow)
            Tx_list.append(Tx)
            Ty_list.append(Ty)

            # cleanup
            del mov_t, fix_t, mov_norm_ds, fix_norm_ds, x, theta
            torch.cuda.empty_cache()

        warped_all = torch.stack(warped_list, dim=-1)
        flow_all = torch.stack(flow_list, dim=-1)
        Tx_all = torch.stack(Tx_list, dim=-1)  # (B,1,D,T)
        Ty_all = torch.stack(Ty_list, dim=-1)
        return warped_all, flow_all, Tx_all, Ty_all

    def training_step(self, batch, batch_idx):
        moving, fixed, mask = batch["moving"], batch["fixed"], batch["mask"]
        warped_all, flow_all, Tx, Ty = self(moving, fixed, mask)
        # Similarity (GNCC + masked L2 + masked L1)
        loss_lncc = global_ncc_loss(warped_all, fixed, mask)  # negative → lower is better
        loss_l2 = l2_loss(warped_all, fixed, mask)

        # Weighted combination
        loss_sim = 0.2 * loss_l2 + 1.0 * loss_lncc

        # Rigid smoothness regularizer
        loss_reg = rigid_smoothness(Tx, Ty)

        # Final composite loss
        loss = loss_sim + loss_reg

        # Logging
        B = moving.size(0)
        self.log("train_loss", loss, prog_bar=True, batch_size=B)
        self.log("train_gncc_loss", loss_lncc, batch_size=B)
        self.log("train_l2_loss", loss_l2, batch_size=B)
        self.log("train_rigid_reg", loss_reg, batch_size=B)
        torch.cuda.empty_cache()
        return loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            moving, fixed, mask = batch["moving"], batch["fixed"], batch["mask"]
            warped_all, flow_all, Tx, Ty = self(moving, fixed, mask)
            loss_lncc = global_ncc_loss(warped_all, fixed, mask)  # negative → lower is better
            loss_l2 = l2_loss(warped_all, fixed, mask)
            loss_sim = 0.2 * loss_l2 + 1.0 * loss_lncc

            loss_reg = rigid_smoothness(Tx, Ty)
            loss = loss_sim + loss_reg

            B = moving.size(0)
            self.log("val_loss", loss, prog_bar=True, batch_size=B)
            self.log("val_gncc_loss", loss_lncc, batch_size=B)
            self.log("val_l2_loss", loss_l2, batch_size=B)
            self.log("val_rigid_reg", loss_reg, batch_size=B)
            torch.cuda.empty_cache()
            return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=15
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }

# -----------------------------
# Parse CLI args
# -----------------------------
if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python moco_main.py <base_path> <data_path> <run_name1> [<run_name2>]")
        sys.exit(1)

    base_path = sys.argv[1]  # e.g. /home/.../moco_dmri/
    data_path = sys.argv[2]  # e.g. /home/.../prepared/dmri_dataset
    order_execution_1 = sys.argv[3]  # run name e.g. yyyymmdd_order_voxelmorph_best-weighted
    order_execution_2 = sys.argv[4] if len(sys.argv) > 4 else None  # format like order_execution_1

    json_path = os.path.join(data_path, "dataset.json")

    print("Base path   :", base_path)
    print("Data folder :", data_path)
    print("JSON path   :", json_path)

    # -----------------------------
    # Training setup
    # -----------------------------
    num_epochs = 200
    batch_size = 1
    lr = 1e-4
    num_workers = 8

    # -----------------------------
    # callbacks
    # -----------------------------
    ckpt_dir = os.path.join(base_path, 'trained_weights')
    os.makedirs(ckpt_dir, exist_ok=True)

    if order_execution_2:
        pretrained_ckpt = os.path.join(ckpt_dir, f"{order_execution_1}.ckpt")
        ckpt_name = f"{order_execution_2}"
    else:
        pretrained_ckpt = None
        ckpt_name = f"{order_execution_1}"

    checkpoint_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename=ckpt_name,
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_weights_only=False
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    early_stop = EarlyStopping(monitor="val_loss", mode="min", patience=25, verbose=True)

    # Load only model weights (use checkpoint as initialization for a new run)
    if pretrained_ckpt is not None and os.path.exists(pretrained_ckpt):
        print(f"Starting NEW training, initialized from: {pretrained_ckpt}")
        model = DenseRigidReg.load_from_checkpoint(pretrained_ckpt, lr=lr)
    else:
        print("No pretrained checkpoint found, training from scratch.")
        model = DenseRigidReg(lr=lr)

    # -----------------------------
    # trainer
    # -----------------------------
    import wandb
    from pytorch_lightning.loggers import WandbLogger

    wandb.login()
    wandb_logger = WandbLogger(project="moco-dmri", name=f"{order_execution_1}")

    # If continue with the pretrained_ckpt, resume logs to the same wandb run
    # wandb_logger = WandbLogger(project="moco-dmri", name=f"{order_execution_2}", id="tia0y7s9", resume="must")

    wandb_config = wandb_logger.experiment.config
    wandb_config.update({
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "lr": lr,
        "data_path": data_path
    }, allow_val_change=True)

    dm = DataModule(json_path=json_path, data_dir=data_path, batch_size=batch_size, num_workers=num_workers)
    gc.collect()
    torch.cuda.empty_cache()
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        logger=wandb_logger,
        callbacks=[checkpoint_cb, lr_monitor, early_stop],
        accelerator="gpu",
        devices=1,
        precision="16-mixed",
        gradient_clip_val=1.0,
        log_every_n_steps=25,
        deterministic=False,
        enable_progress_bar=True,
        enable_model_summary=True,
        enable_checkpointing=True
    )

    # -----------------------------
    # Fit model
    # -----------------------------
    # Load full checkpoint (resume training exactly where it left off)
    # if os.path.exists(pretrained_ckpt):
    #     print(f"Resuming training from: {pretrained_ckpt}")
    #     trainer.fit(model, datamodule=dm, ckpt_path=pretrained_ckpt)
    # else:
    #     print("No checkpoint found, training from scratch.")
    trainer.fit(model, datamodule=dm)

    # -----------------------------
    # Best checkpoint info
    # -----------------------------
    print("Best checkpoint path:", checkpoint_cb.best_model_path)
    if checkpoint_cb.best_model_score is not None:
        best_val_loss = checkpoint_cb.best_model_score.item()
        print("Best val_loss:", best_val_loss)
    else:
        best_val_loss = None
        print("No checkpoint was saved yet.")

    wandb_logger.log_metrics({
        "best_val_loss": best_val_loss,
        "best_checkpoint_path": checkpoint_cb.best_model_path
    })

    end_time = time.ctime()
    print("End at:", end_time)