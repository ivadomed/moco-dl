#!/usr/bin/env python3
"""
Composite loss for moco-dl training.

Components
----------
1. Image similarity: masked local NCC (LNCC) between warped moving and fixed.
   Local NCC is computed in 9x9x3 patches restricted to the spinal-cord mask.
   This replaces the *global* NCC in the original moco_main.py, which was
   dominated by background voxels rather than the cord ROI.

2. Supervised regression: MSE between predicted (Tx, Ty, Theta) and the
   ground-truth values from the augmenter. This is the strongest learning
   signal in the new pipeline — the original code threw away the GT shifts.
   Note that for cases where the augmenter applied no motion, GT is zero,
   so this term explicitly teaches "predict zero when nothing moved."

3. Smoothness regularization: L1 on differences along z and t. Discourages
   jitter in predictions across adjacent slices and timepoints.

All terms are computed at full timeseries resolution. The forward pass loops
over t for memory reasons; the loss is summed across t and averaged.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _normalize_in_mask(x: torch.Tensor, mask: torch.Tensor,
                       p_lo: float = 0.01, p_hi: float = 0.99,
                       eps: float = 1e-6) -> torch.Tensor:
    """
    Percentile-normalize x to [0, 1] using percentiles computed inside mask.

    x:    (B, 1, H, W, D) or (B, 1, H, W, D, T)
    mask: (B, 1, H, W, D) — broadcast across T if needed.
    """
    m = (mask > 0).float()
    if x.ndim == 6:
        m = m.unsqueeze(-1).expand_as(x)
    vals = x[m.bool()]
    if vals.numel() < 16:
        return x  # mask too small to normalize meaningfully
    lo = torch.quantile(vals, p_lo)
    hi = torch.quantile(vals, p_hi)
    return ((x - lo) / (hi - lo + eps)).clamp(0, 1)


def local_ncc_3d(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor,
                 win: tuple[int, int, int] = (9, 9, 3),
                 eps: float = 1e-5) -> torch.Tensor:
    """
    Local NCC over 3D patches, masked.

    a, b: (B, 1, H, W, D) — already on similar intensity scale (use _normalize_in_mask)
    mask: (B, 1, H, W, D)
    Returns the *negative* mean LNCC inside the mask, so that lower = better
    alignment (suitable as a loss term).

    Uses one 3D conv per moment to compute window means & cross-products.

    NOTE: the math here can underflow in FP16 (var_a * var_b can round to 0,
    yielding inf after sqrt → NaN downstream). We force the whole computation
    to FP32 even under autocast.
    """
    kH, kW, kD = win
    pad = (kD // 2, kD // 2, kW // 2, kW // 2, kH // 2, kH // 2)

    # Run the entire LNCC in FP32 regardless of mixed-precision context
    with torch.amp.autocast(device_type=a.device.type, enabled=False):
        a32 = a.float()
        b32 = b.float()

        kernel = torch.ones(1, 1, kH, kW, kD, device=a32.device, dtype=a32.dtype)
        n = float(kH * kW * kD)

        a_p = F.pad(a32, pad, mode="reflect")
        b_p = F.pad(b32, pad, mode="reflect")
        a_sum = F.conv3d(a_p, kernel)
        b_sum = F.conv3d(b_p, kernel)
        ab_sum = F.conv3d(a_p * b_p, kernel)
        a2_sum = F.conv3d(a_p * a_p, kernel)
        b2_sum = F.conv3d(b_p * b_p, kernel)

        a_mean = a_sum / n
        b_mean = b_sum / n
        cov = ab_sum / n - a_mean * b_mean
        # Variance can be slightly negative due to floating-point cancellation;
        # clamp BEFORE multiplying so the product stays positive.
        var_a = (a2_sum / n - a_mean * a_mean).clamp_min(eps)
        var_b = (b2_sum / n - b_mean * b_mean).clamp_min(eps)
        ncc = cov / torch.sqrt(var_a * var_b)

        # Masked mean
        m = (mask > 0).float()
        denom = m.sum().clamp_min(1.0)
        masked_ncc = (ncc * m).sum() / denom
        return -masked_ncc  # lower is better


def _get_fixed_t(fixed: torch.Tensor, t: int) -> torch.Tensor:
    """Return (B,1,H,W,D) for fixed at timepoint t — handles 5D and 6D fixed."""
    if fixed.ndim == 5:
        return fixed
    if fixed.ndim == 6:
        return fixed[..., t].contiguous()
    raise ValueError(f"Unexpected fixed.ndim={fixed.ndim}")


# ---------------------------------------------------------------------------
# Composite loss
# ---------------------------------------------------------------------------

class MocoLoss(nn.Module):
    """
    Composite training loss with three weighted terms.

    Parameters
    ----------
    w_sim : float
        Weight on image similarity (masked LNCC) between warped moving and fixed.
    w_reg : float
        Weight on supervised regression of (Tx, Ty, Theta) against GT.
        Set to 0 to fall back to fully unsupervised training.
    w_smooth : float
        Weight on L1 smoothness of predictions across z and t.
    lncc_win : tuple[int, int, int]
        Local-NCC patch size (H, W, D).
    """

    def __init__(self,
                 w_sim: float = 1.0,
                 w_reg: float = 1.0,
                 w_smooth: float = 0.1,
                 lncc_win: tuple[int, int, int] = (9, 9, 3)):
        super().__init__()
        self.w_sim = w_sim
        self.w_reg = w_reg
        self.w_smooth = w_smooth
        self.lncc_win = lncc_win

    def forward(self,
                warped: torch.Tensor,         # (B, 1, H, W, D, T)
                fixed: torch.Tensor,          # (B, 1, H, W, D) or (B, 1, H, W, D, T)
                mask: torch.Tensor,           # (B, 1, H, W, D)
                Tx_pred: torch.Tensor,        # (B, D, T)
                Ty_pred: torch.Tensor,        # (B, D, T)
                Theta_pred: torch.Tensor,     # (B, D, T) in degrees
                Tx_gt: torch.Tensor,          # (B, D, T)
                Ty_gt: torch.Tensor,          # (B, D, T)
                Theta_gt: torch.Tensor | None = None,  # (B, D, T) or None
                ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Returns (total_loss, components) where components is a dict suitable
        for logging.
        """
        B, _, H, W, D, T = warped.shape

        # ---- 1. Image similarity (LNCC, masked) ----
        sim_per_t = []
        for t in range(T):
            w_t = warped[..., t]
            f_t = _get_fixed_t(fixed, t)
            w_n = _normalize_in_mask(w_t, mask)
            f_n = _normalize_in_mask(f_t, mask)
            sim_per_t.append(local_ncc_3d(w_n, f_n, mask, win=self.lncc_win))
        loss_sim = torch.stack(sim_per_t).mean()

        # ---- 2. Supervised regression on shifts ----
        loss_reg_xy = F.mse_loss(Tx_pred, Tx_gt) + F.mse_loss(Ty_pred, Ty_gt)
        if Theta_gt is not None:
            loss_reg_theta = F.mse_loss(Theta_pred, Theta_gt)
        else:
            # No theta GT — penalize large theta predictions only weakly,
            # so the network is still allowed to use the rotation degree
            # of freedom but not encouraged to.
            loss_reg_theta = (Theta_pred ** 2).mean() * 0.01
        loss_reg = loss_reg_xy + loss_reg_theta

        # ---- 3. Smoothness across z and t (L1) ----
        def _l1_diff(p: torch.Tensor) -> torch.Tensor:
            dz = (p[:, 1:, :] - p[:, :-1, :]).abs().mean()
            dt = (p[:, :, 1:] - p[:, :, :-1]).abs().mean()
            return dz + dt

        loss_smooth = (_l1_diff(Tx_pred) + _l1_diff(Ty_pred) +
                       0.1 * _l1_diff(Theta_pred))

        total = (self.w_sim * loss_sim +
                 self.w_reg * loss_reg +
                 self.w_smooth * loss_smooth)

        components = {
            "loss/total": total.detach(),
            "loss/sim_lncc": loss_sim.detach(),
            "loss/reg_xy": loss_reg_xy.detach(),
            "loss/reg_theta": loss_reg_theta.detach() if isinstance(loss_reg_theta, torch.Tensor) else torch.tensor(0.0),
            "loss/smooth": loss_smooth.detach(),
        }
        return total, components


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)
    B, H, W, D, T = 1, 32, 32, 6, 4
    warped = torch.randn(B, 1, H, W, D, T)
    fixed = torch.randn(B, 1, H, W, D)
    mask = torch.zeros(B, 1, H, W, D)
    mask[..., 10:22, 10:22, 1:5] = 1.0

    Tx_p = torch.randn(B, D, T) * 0.1
    Ty_p = torch.randn(B, D, T) * 0.1
    Th_p = torch.randn(B, D, T) * 0.1
    Tx_g = torch.randn(B, D, T) * 0.5
    Ty_g = torch.randn(B, D, T) * 0.5
    Th_g = torch.randn(B, D, T) * 0.5

    loss_fn = MocoLoss()
    total, comps = loss_fn(warped, fixed, mask, Tx_p, Ty_p, Th_p, Tx_g, Ty_g, Th_g)
    print(f"Total: {total.item():.4f}")
    for k, v in comps.items():
        print(f"  {k}: {v.item():.4f}")
    print(f"\nGradient flow check:")
    Tx_p.requires_grad_(True)
    total, _ = loss_fn(warped, fixed, mask, Tx_p, Ty_p, Th_p, Tx_g, Ty_g, Th_g)
    total.backward()
    print(f"  Tx_p.grad is not None: {Tx_p.grad is not None}")
    print(f"  Tx_p.grad mean abs: {Tx_p.grad.abs().mean():.6f}")
    print("\nSelf-test passed.")
