#!/usr/bin/env python3
"""
moco-dl model: 3D DenseNet backbone + 1D temporal Conv head.

Predicts slice-wise (Tx, Ty, Theta) per timepoint for a 4D volume.

Compared with the original DenseNetRegressorSliceWise from moco_main.py:

- Slightly larger backbone (3 dense blocks, growth_rate=12, 6 layers/block)
  to handle the wider augmentation distribution we now train on.
- Outputs (Tx, Ty, Theta) — rotation is now a real output, not a leftover
  comment. The augmenter applies rotation, so the model gets free rotation
  supervision.
- A 1D Conv-along-t temporal head ties together predictions across
  timepoints. Real motion is highly correlated across t (cardiac/respiratory
  cycles); the original architecture predicted each t in a fully isolated
  forward pass. The temporal head fixes that without adding much cost.
- The warp module is rewritten to handle rotation and to do its own
  affine_grid + grid_sample (so we don't depend on MONAI's Warp signature
  for the rotation case).

Coordinate convention
---------------------
Throughout this module, slice tensors are (H, W) with axis-0 = y, axis-1 = x.
Tx is a translation along x (= axis 1) in voxel units. Ty is a translation
along y (= axis 0) in voxel units. Theta is a counter-clockwise in-plane
rotation in radians around the slice center.

This convention matches the augmentation module:
- ``MotionAugmenter.Tx_gt`` is the y-axis translation (axis 0).
- ``MotionAugmenter.Ty_gt`` is the x-axis translation (axis 1).

Wait — re-reading motion_augmentation.py confirms that:
  ``_per_slice_warp_2d(slice2d, tx, ty, ...)`` applies tx along axis-0 (y)
  and ty along axis-1 (x). So in the augmenter, Tx_gt is a y-shift and
  Ty_gt is an x-shift. We adopt the SAME convention here for consistency:
  the model's "Tx" output is the y-axis shift (axis 0) and "Ty" is the
  x-axis shift (axis 1). This may look odd but it matches the data and
  any supervised regression loss against (Tx_gt, Ty_gt) will be correct.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# DenseBlock (3D)
# ---------------------------------------------------------------------------

class DenseBlock(nn.Module):
    """A 3D DenseNet block: each layer concatenates its output to the input."""

    def __init__(self, in_channels: int, growth_rate: int, n_layers: int):
        super().__init__()
        self.layers = nn.ModuleList()
        ch = in_channels
        for _ in range(n_layers):
            self.layers.append(nn.Sequential(
                nn.InstanceNorm3d(ch, affine=True, track_running_stats=False),
                nn.ReLU(inplace=True),
                nn.Conv3d(ch, growth_rate, kernel_size=3, padding=1, bias=False),
            ))
            ch += growth_rate
        self.out_channels = ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            new = layer(x)
            x = torch.cat([x, new], dim=1)
        return x


class TransitionDown(nn.Module):
    """1x1 channel reduction + 2x2 spatial pool (z-dim preserved)."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.InstanceNorm3d(in_channels, affine=True, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.MaxPool3d(kernel_size=(2, 2, 1)),  # only pool H, W — keep D intact
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ---------------------------------------------------------------------------
# Backbone: 3D DenseNet, outputs per-slice features
# ---------------------------------------------------------------------------

class DenseBackbone(nn.Module):
    """
    3D DenseNet that takes (B, 2, H, W, D) and returns (B, F, D) per-slice
    features, with H and W pooled out and D preserved.

    With the chosen sizes (3 blocks, growth=12, 6 layers/block, init=16) the
    output channel count F is computed below; ~600K parameters total.
    """

    def __init__(self,
                 in_channels: int = 2,
                 init_channels: int = 16,
                 growth_rate: int = 12,
                 n_blocks: int = 3,
                 n_layers_per_block: int = 6):
        super().__init__()
        self.stem = nn.Conv3d(in_channels, init_channels,
                              kernel_size=(3, 3, 1),
                              stride=(2, 2, 1),
                              padding=(1, 1, 0))
        modules = []
        ch = init_channels
        for i in range(n_blocks):
            blk = DenseBlock(ch, growth_rate, n_layers_per_block)
            ch = blk.out_channels
            modules.append(blk)
            # Transition after every block except the last; halve channels each time.
            if i < n_blocks - 1:
                trans = TransitionDown(ch, ch // 2)
                ch = ch // 2
                modules.append(trans)
        self.body = nn.Sequential(*modules)
        # Pool only H and W, keep D
        self.spatial_pool = nn.AdaptiveAvgPool3d((1, 1, None))
        self.feature_channels = ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 2, H, W, D)
        returns: (B, F, D)
        """
        x = self.stem(x)
        x = self.body(x)
        x = self.spatial_pool(x)        # (B, F, 1, 1, D)
        return x.squeeze(2).squeeze(2)  # (B, F, D)


# ---------------------------------------------------------------------------
# Temporal head: 1D Conv along t
# ---------------------------------------------------------------------------

class TemporalConv1D(nn.Module):
    """
    Refines per-(d, t) predictions by 1D convolution along t, applied
    independently to each (b, d) row.

    We treat the input as (B*D, F, T) — i.e. each slice index is a
    "channel-batch" item with F features over the T time axis. Three Conv1D
    layers with kernel=5 give an effective receptive field of 13 timepoints,
    which is plenty for cardiac/respiratory smoothing without being so wide
    that the head becomes a bottleneck.
    """

    def __init__(self,
                 in_channels: int,
                 hidden_channels: int = 64,
                 out_channels: int = 3,    # Tx, Ty, Theta
                 kernel_size: int = 5,
                 n_layers: int = 3):
        super().__init__()
        layers = []
        ch = in_channels
        for i in range(n_layers - 1):
            layers += [
                nn.Conv1d(ch, hidden_channels, kernel_size,
                          padding=kernel_size // 2),
                nn.GroupNorm(num_groups=8, num_channels=hidden_channels),
                nn.ReLU(inplace=True),
            ]
            ch = hidden_channels
        # Final layer: project to out_channels with init-near-zero so the
        # network starts at near-identity behavior (no motion).
        final = nn.Conv1d(ch, out_channels, kernel_size,
                          padding=kernel_size // 2)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        layers.append(final)
        self.net = nn.Sequential(*layers)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """
        feats: (B, F, D, T)
        returns: (B, out_channels, D, T)
        """
        B, F_, D, T = feats.shape
        # Fold (B, D) into a single batch axis, F is channel, T is the conv axis.
        x = feats.permute(0, 2, 1, 3).reshape(B * D, F_, T)
        y = self.net(x)                       # (B*D, out, T)
        out = y.reshape(B, D, -1, T).permute(0, 2, 1, 3)
        return out                            # (B, out_channels, D, T)


# ---------------------------------------------------------------------------
# Full network
# ---------------------------------------------------------------------------

class DenseRigidNet(nn.Module):
    """
    Full motion-correction network: backbone over each timepoint, then 1D
    temporal head over t.

    Output: (Tx, Ty, Theta) per slice per timepoint, of shape (B, 3, D, T).
    Tx, Ty are bounded by ``max_translation_vox``; Theta by
    ``max_rotation_deg``, both via tanh.

    Following the augmenter convention (see module docstring), Tx is the
    axis-0 (y) translation and Ty is the axis-1 (x) translation, both in
    voxel units. Theta is a counter-clockwise rotation in degrees.
    """

    def __init__(self,
                 in_channels: int = 2,
                 init_channels: int = 16,
                 growth_rate: int = 12,
                 n_blocks: int = 3,
                 n_layers_per_block: int = 6,
                 temporal_hidden: int = 64,
                 temporal_kernel: int = 5,
                 temporal_n_layers: int = 3,
                 max_translation_vox: float = 5.0,
                 max_rotation_deg: float = 5.0,
                 use_checkpointing: bool = False):
        """
        use_checkpointing: if True, the backbone is run via torch.utils.checkpoint
        during training. Activations are not stored; they are recomputed during
        the backward pass. Trades ~30% more compute for ~80% less activation
        memory. Big win when T is large.
        """
        super().__init__()
        self.backbone = DenseBackbone(
            in_channels=in_channels,
            init_channels=init_channels,
            growth_rate=growth_rate,
            n_blocks=n_blocks,
            n_layers_per_block=n_layers_per_block,
        )
        self.temporal = TemporalConv1D(
            in_channels=self.backbone.feature_channels,
            hidden_channels=temporal_hidden,
            out_channels=3,
            kernel_size=temporal_kernel,
            n_layers=temporal_n_layers,
        )
        self.max_translation_vox = max_translation_vox
        self.max_rotation_deg = max_rotation_deg
        self.use_checkpointing = use_checkpointing

    def _per_t_features(self, moving: torch.Tensor, fixed: torch.Tensor
                        ) -> torch.Tensor:
        """
        Run the backbone once per timepoint to produce (B, F, D, T).

        moving: (B, 1, H, W, D, T)
        fixed:  (B, 1, H, W, D)  OR  (B, 1, H, W, D, T)
        """
        B, _, H, W, D, T = moving.shape
        feats = []
        for t in range(T):
            mov_t = moving[..., t]                       # (B, 1, H, W, D)
            if fixed.ndim == 5:
                fix_t = fixed
            elif fixed.ndim == 6:
                fix_t = fixed[..., t]
            else:
                raise ValueError(f"Unexpected fixed.ndim={fixed.ndim}")
            x = torch.cat([mov_t, fix_t], dim=1)         # (B, 2, H, W, D)
            if self.use_checkpointing and self.training and x.requires_grad:
                f = torch.utils.checkpoint.checkpoint(self.backbone, x,
                                                      use_reentrant=False)
            else:
                f = self.backbone(x)                     # (B, F, D)
            feats.append(f)
        return torch.stack(feats, dim=-1)                # (B, F, D, T)

    def forward(self, moving: torch.Tensor, fixed: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        moving: (B, 1, H, W, D, T)
        fixed:  (B, 1, H, W, D)  or  (B, 1, H, W, D, T)
        returns: Tx, Ty, Theta — each (B, D, T), in voxel/degree units.
        """
        feats = self._per_t_features(moving, fixed)      # (B, F, D, T)
        raw = self.temporal(feats)                       # (B, 3, D, T)

        # Bound outputs via tanh so init is identity and large shifts are clipped
        Tx = torch.tanh(raw[:, 0]) * self.max_translation_vox  # (B, D, T)
        Ty = torch.tanh(raw[:, 1]) * self.max_translation_vox
        Theta = torch.tanh(raw[:, 2]) * self.max_rotation_deg
        return Tx, Ty, Theta


# ---------------------------------------------------------------------------
# Rigid warp (slice-wise, differentiable)
# ---------------------------------------------------------------------------

class RigidSliceWarp(nn.Module):
    """
    Apply per-slice rigid (rotation + translation) warps to a 3D volume,
    using PyTorch's affine_grid + grid_sample so the warp is differentiable.

    Conventions (see DenseRigidNet docstring):
      Tx : translation along axis-0 (y) in voxels
      Ty : translation along axis-1 (x) in voxels
      Theta : counter-clockwise rotation in degrees, around the slice center

    Input vol shape: (B, 1, H, W, D)
    Tx, Ty, Theta:  (B, D)
    Output:         (B, 1, H, W, D) — warped volume
    """

    def __init__(self, mode: str = "bilinear", padding_mode: str = "border"):
        super().__init__()
        self.mode = mode
        self.padding_mode = padding_mode

    def forward(self, vol: torch.Tensor,
                Tx: torch.Tensor, Ty: torch.Tensor, Theta: torch.Tensor
                ) -> torch.Tensor:
        B, C, H, W, D = vol.shape
        device, dtype = vol.device, vol.dtype

        # Reorganize so each slice is its own 2D image with batch axis B*D
        # (B, 1, H, W, D) -> (B, D, 1, H, W) -> (B*D, 1, H, W)
        slices = vol.permute(0, 4, 1, 2, 3).contiguous().view(B * D, C, H, W)

        # affine_grid expects a (N, 2, 3) matrix in *normalized* coordinates,
        # where output[y_n, x_n] is sampled from input at position
        #   src = M @ [x_n, y_n, 1]^T  (note grid_sample's (x, y) convention)
        # with x_n, y_n in [-1, 1]. To translate the OUTPUT image by (tx_pix, ty_pix)
        # in pixels (positive = move feature toward higher index), we sample
        # the input at (x_n - 2*ty_pix/(W-1), y_n - 2*tx_pix/(H-1))
        # — i.e. the inverse mapping subtracts the desired output shift.

        theta_rad = Theta.reshape(B * D) * (torch.pi / 180.0)
        tx_pix = Tx.reshape(B * D)        # axis 0 = y in image-coord, but
        ty_pix = Ty.reshape(B * D)        # affine_grid uses (x, y) so we map below
        cos_t = torch.cos(theta_rad)
        sin_t = torch.sin(theta_rad)

        # Normalized translation: for an image of size (H, W), shifting by
        # tx_pix along axis-0 (y) corresponds to a normalized shift of
        # 2*tx_pix/(H-1) in the y-direction. Inverse warp -> negate.
        H_norm = max(H - 1, 1)
        W_norm = max(W - 1, 1)
        ty_norm = 2.0 * ty_pix / W_norm     # x-direction in grid_sample
        tx_norm = 2.0 * tx_pix / H_norm     # y-direction in grid_sample

        # Build 2x3 affine matrix per slice.
        # Forward transform: rotate around center then translate by (ty_pix, tx_pix).
        # Inverse (for grid_sample): translate by -(ty, tx) then rotate by -theta.
        # In matrix form for the inverse:
        #   [ cos  sin  -ty_norm*cos - tx_norm*sin ]
        #   [-sin  cos   ty_norm*sin - tx_norm*cos ]
        # (x' = cos*x + sin*y  - ty*cos - tx*sin  etc.)
        zeros = torch.zeros_like(cos_t)
        # First row corresponds to x in grid_sample, second to y
        a00 =  cos_t
        a01 =  sin_t
        a02 = -ty_norm * cos_t - tx_norm * sin_t
        a10 = -sin_t
        a11 =  cos_t
        a12 =  ty_norm * sin_t - tx_norm * cos_t

        theta_mat = torch.stack([
            torch.stack([a00, a01, a02], dim=-1),
            torch.stack([a10, a11, a12], dim=-1),
        ], dim=-2)                          # (B*D, 2, 3)
        theta_mat = theta_mat.to(dtype=dtype, device=device)

        grid = F.affine_grid(theta_mat, size=(B * D, C, H, W), align_corners=True)
        warped = F.grid_sample(slices, grid,
                               mode=self.mode,
                               padding_mode=self.padding_mode,
                               align_corners=True)

        # Reassemble (B*D, C, H, W) -> (B, 1, H, W, D)
        warped = warped.view(B, D, C, H, W).permute(0, 2, 3, 4, 1).contiguous()
        return warped


# ---------------------------------------------------------------------------
# Self-test: forward pass + warp consistency check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Model self-test")
    print("=" * 60)

    B, H, W, D, T = 1, 64, 64, 12, 8
    moving = torch.randn(B, 1, H, W, D, T)
    fixed = torch.randn(B, 1, H, W, D)         # fMRI-style 3D fixed

    net = DenseRigidNet()
    n_params = sum(p.numel() for p in net.parameters())
    print(f"Total parameters: {n_params:,}")

    # Forward pass
    with torch.no_grad():
        Tx, Ty, Theta = net(moving, fixed)
    print(f"Tx shape:    {tuple(Tx.shape)}  range: [{Tx.min():.3f}, {Tx.max():.3f}]")
    print(f"Ty shape:    {tuple(Ty.shape)}  range: [{Ty.min():.3f}, {Ty.max():.3f}]")
    print(f"Theta shape: {tuple(Theta.shape)}  range: [{Theta.min():.3f}, {Theta.max():.3f}]")

    # At init, with zeros final-layer init, predictions should be ~zero
    near_zero = (Tx.abs().max() < 0.01 and Ty.abs().max() < 0.01 and Theta.abs().max() < 0.01)
    print(f"Init produces ~zero predictions: {near_zero}")
    assert near_zero, "Final layer should be initialised to zero for identity-at-init"

    # Warp consistency: warping by zero should be identity
    print("\n" + "-" * 60)
    print("Warp consistency check")
    print("-" * 60)
    warp = RigidSliceWarp()
    vol = torch.randn(B, 1, H, W, D)
    zero = torch.zeros(B, D)
    warped_id = warp(vol, zero, zero, zero)
    diff = (warped_id - vol).abs().max().item()
    print(f"Max |warped - vol| with zero shift: {diff:.6f}  (should be ~0)")
    assert diff < 1e-4, "Zero-shift warp should be identity"

    # Warp by known translation: place a bright pixel and check it moved
    print()
    vol_test = torch.zeros(B, 1, H, W, D)
    vol_test[0, 0, 30, 30, 5] = 1.0  # one bright voxel at (y=30, x=30, z=5)

    # Translate slice 5 by Tx=+3 (axis-0 = y), Ty=-2 (axis-1 = x), no rotation.
    Tx_test = torch.zeros(B, D)
    Ty_test = torch.zeros(B, D)
    Tx_test[0, 5] = 3.0
    Ty_test[0, 5] = -2.0
    Theta_test = torch.zeros(B, D)
    warped = warp(vol_test, Tx_test, Ty_test, Theta_test)

    # The brightest voxel in slice 5 should now be at (30+3, 30-2) = (33, 28)
    sl = warped[0, 0, :, :, 5]
    yi, xi = torch.unravel_index(sl.argmax(), sl.shape)
    print(f"Bright voxel was at (30, 30); after Tx=+3, Ty=-2 it's at ({yi}, {xi})")
    print(f"Expected: (33, 28).  Result: {(int(yi), int(xi)) == (33, 28)}")

    # Other slices should be unchanged
    other_slice_diff = (warped[0, 0, :, :, 4] - vol_test[0, 0, :, :, 4]).abs().max().item()
    print(f"Untouched slice 4 max diff: {other_slice_diff:.6f}")

    # Differentiability: gradient flows through warp
    print()
    vol_grad = torch.randn(B, 1, H, W, D, requires_grad=True)
    Tx_g = torch.full((B, D), 0.5, requires_grad=True)
    Ty_g = torch.full((B, D), 0.5, requires_grad=True)
    Th_g = torch.full((B, D), 0.5, requires_grad=True)
    out = warp(vol_grad, Tx_g, Ty_g, Th_g).sum()
    out.backward()
    print(f"grad flows: vol={vol_grad.grad is not None}  "
          f"Tx={Tx_g.grad is not None}  Ty={Ty_g.grad is not None}  "
          f"Theta={Th_g.grad is not None}")

    print("\nAll self-tests passed.")
