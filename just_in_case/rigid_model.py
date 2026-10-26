"""
Rigid (rotation + translation only, no scale/shear) 2D registration network.

Predicts a 3-parameter transform (theta, tx, ty) from a moving/fixed image
pair and applies it via torch's affine_grid + grid_sample -- a classic spatial
transformer network (STN) constrained to the rigid subgroup of 2D affine
transforms. No dense per-pixel deformation field is produced, so there is no
local warping: every pixel moves under the same global rotation+translation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RigidRegistrationNet(nn.Module):
    def __init__(self, in_channels: int = 2, base_channels: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, stride=2, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, stride=2, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(base_channels * 4, base_channels * 8, 3, stride=2, padding=1), nn.LeakyReLU(0.2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(base_channels * 8, 3)  # theta, tx, ty

        # Initialize near identity (zero rotation, zero translation) -- standard
        # STN practice so training starts from "do nothing" rather than a random
        # transform, which stabilizes early optimization.
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def predict_params(self, moving: torch.Tensor, fixed: torch.Tensor) -> torch.Tensor:
        """Returns raw (B, 3) params: theta (radians), tx, ty (normalized [-1, 1] coords)."""
        x = torch.cat([moving, fixed], dim=1)
        feat = self.encoder(x)
        feat = self.pool(feat).flatten(1)
        return self.fc(feat)

    def forward(self, moving: torch.Tensor, fixed: torch.Tensor):
        """
        Returns (warped_moving, affine_matrix) where affine_matrix is (B, 2, 3),
        matching the convention expected by F.affine_grid.
        """
        params = self.predict_params(moving, fixed)
        affine_matrix = params_to_affine_matrix(params)
        warped = warp_with_affine(moving, affine_matrix)
        return warped, affine_matrix


def params_to_affine_matrix(params: torch.Tensor) -> torch.Tensor:
    """
    Build a (B, 2, 3) rigid affine matrix from (theta, tx, ty) params.
    Only rotation + translation -- no scale or shear terms, so the transform
    is structurally constrained to be rigid regardless of what the network learns.
    """
    theta, tx, ty = params[:, 0], params[:, 1], params[:, 2]
    cos_t, sin_t = torch.cos(theta), torch.sin(theta)

    row0 = torch.stack([cos_t, -sin_t, tx], dim=1)
    row1 = torch.stack([sin_t, cos_t, ty], dim=1)
    return torch.stack([row0, row1], dim=1)  # (B, 2, 3)


def warp_with_affine(image: torch.Tensor, affine_matrix: torch.Tensor,
                      mode: str = "bilinear") -> torch.Tensor:
    """
    Apply a (B, 2, 3) affine matrix to a (B, C, H, W) image via affine_grid + grid_sample.
    mode: 'bilinear' (default, required for training -- needs a non-zero gradient
    almost everywhere) or 'nearest' (sharper, no blending -- an inference-only option
    since it has ~zero gradient and would break backprop during training).
    """
    grid = F.affine_grid(affine_matrix, image.shape, align_corners=False)
    return F.grid_sample(image, grid, mode=mode, padding_mode="border", align_corners=False)
