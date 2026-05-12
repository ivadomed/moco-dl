#!/usr/bin/env python3
"""
On-the-fly motion + intensity augmentation for moco-dl training.

Goals
-----
The legacy ``augmentation.py`` pre-generates motion-augmented NIfTI files
once, with i.i.d. uniform per-slice translations and nearest-neighbor
interpolation. That distribution does not look like real spinal-cord motion
(which is smooth across adjacent slices, smooth across time, heavy-tailed in
amplitude, and comes bundled with rotation, through-plane motion, and
intensity perturbations). This module replaces it with a richer, on-the-fly
augmentation that can be applied inside the DataLoader so every epoch sees
fresh motion patterns.

What it produces
----------------
Given a clean 4D volume ``vol`` of shape ``(H, W, D, T)`` and a 3D mask of
shape ``(H, W, D)``, ``MotionAugmenter.__call__`` returns:

    moving      : (H, W, D, T) float32  — vol with simulated motion + intensity
    Tx_gt       : (D, T)        float32 — ground-truth x-translation per (z, t)
    Ty_gt       : (D, T)        float32 — ground-truth y-translation per (z, t)

Tx_gt / Ty_gt are exactly the in-plane translations applied to each slice
at each timepoint. They can be used either purely for evaluation or for an
explicit supervised regression loss against the network's predictions.

Note: rotation and through-plane motion are also applied to the image, but
they are not exposed in the GT — the current model only predicts (Tx, Ty)
per slice. If the model is later extended to predict rotation as well, the
returned dict can be expanded.

Design notes
------------
- Motion fields are built on a coarse (z, t) control grid and upsampled with
  cubic interpolation, then jittered with small i.i.d. Gaussian noise. This
  yields spatial/temporal coherence (real motion) plus realistic local noise.
- Per-case amplitude is sampled from a half-normal-like prior: most cases
  have small motion, occasional cases have large motion.
- Interpolation is bilinear (consistent with MONAI's Warp used at training
  time). The legacy nearest-neighbor choice was a known source of train/test
  mismatch — see issue #14.
- Intensity transforms (Rician noise, bias field, gamma, slice dropout) are
  applied AFTER motion so the augmentation order matches what really happens
  during acquisition: the body moves, then the scanner samples a slightly
  different field/contrast.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates, gaussian_filter

# ---------------------------------------------------------------------------
# Helpers: smooth random fields
# ---------------------------------------------------------------------------

def _smooth_random_field(D: int, T: int, n_ctrl_z: int, n_ctrl_t: int,
                         rng: np.random.Generator) -> np.ndarray:
    """
    Generate a smooth random field of shape (D, T) by sampling a coarse
    control grid and upsampling with cubic interpolation along both axes.

    The resulting field has standard deviation ~1 (rescale outside).
    """
    # Coarse grid of standard normal samples
    coarse = rng.standard_normal(size=(max(n_ctrl_z, 2), max(n_ctrl_t, 2)))

    # Upsample along z, then along t, using map_coordinates with cubic order
    # (we use map_coordinates twice — separable cubic upsampling).
    z_query = np.linspace(0, coarse.shape[0] - 1, D)
    t_query = np.linspace(0, coarse.shape[1] - 1, T)

    # First interpolate along z for each control-t column
    z_interp = np.empty((D, coarse.shape[1]))
    for j in range(coarse.shape[1]):
        z_interp[:, j] = map_coordinates(coarse[:, j], [z_query], order=3, mode="reflect")

    # Then interpolate along t for each z row
    field = np.empty((D, T))
    for i in range(D):
        field[i, :] = map_coordinates(z_interp[i, :], [t_query], order=3, mode="reflect")

    # Re-normalize to ~unit std (cubic upsampling damps variance a bit)
    s = field.std()
    if s > 1e-8:
        field = field / s
    return field.astype(np.float32)


def _per_slice_warp_2d(slice2d: np.ndarray, tx: float, ty: float, theta_deg: float,
                       order: int = 1, mode: str = "reflect") -> np.ndarray:
    """
    Apply a 2D rigid transform (rotation by theta_deg around image center, then
    translation by (tx, ty)) to a single slice using map_coordinates.

    Parameters
    ----------
    slice2d : (H, W) array
    tx, ty  : translations in voxel units (tx along axis-0, ty along axis-1)
    theta_deg : in-plane rotation in degrees, around image center
    order   : spline order (1 = bilinear)
    """
    H, W = slice2d.shape
    yy, xx = np.indices((H, W), dtype=np.float32)
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0

    # Coordinates relative to center
    yc = yy - cy
    xc = xx - cx

    theta = np.deg2rad(theta_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    # Inverse mapping: where in the source do we sample for each output pixel?
    # Forward transform is rotate then translate; inverse is translate-back
    # then rotate-back.
    src_y = (yc - tx) * cos_t + (xc - ty) * sin_t + cy
    src_x = -(yc - tx) * sin_t + (xc - ty) * cos_t + cx

    coords = np.stack([src_y, src_x], axis=0)
    return map_coordinates(slice2d, coords, order=order, mode=mode, cval=0.0)


# ---------------------------------------------------------------------------
# Main augmenter
# ---------------------------------------------------------------------------

class MotionAugmenter:
    """
    On-the-fly motion + intensity augmentation for slice-wise 4D volumes.

    Applied to ``vol`` of shape (H, W, D, T) — produces a motion-augmented
    copy plus the ground-truth (Tx, Ty) per slice per timepoint.

    Parameters
    ----------
    p_case : float
        Probability that this volume gets ANY motion. With prob (1-p_case)
        the returned moving == vol and Tx_gt = Ty_gt = 0. This is critical
        so the network learns "predict zero when nothing moved" — addresses
        the main symptom of issue #14.
    case_scale_sigma : float
        Per-case amplitude. The case-level scale is sampled from
        |N(0, case_scale_sigma)| + 0.3, so most cases have small motion and
        a long tail of larger motion.
    max_translation_vox : float
        Hard cap on |Tx|, |Ty| after sampling, in voxels.
    max_rotation_deg : float
        Hard cap on |theta|, in degrees.
    max_through_plane_vox : float
        Hard cap on |Tz|, in voxels. Through-plane motion shifts the volume
        along z by a non-integer amount via cubic interpolation. Note this
        is NOT exposed as ground truth (the model doesn't predict it).
    n_ctrl_z, n_ctrl_t : int
        Number of control points in the coarse (z, t) grid that drives the
        smooth motion field. Smaller -> smoother. 4 / 6 are good defaults.
    jitter_sigma_vox : float
        Std of i.i.d. Gaussian jitter added on top of the smooth field, in
        voxels. Adds realistic per-slice variability.
    p_intensity : float
        Probability of applying intensity perturbations (noise, bias, gamma).
    p_dropout : float
        Probability that a moving timepoint has 1-2 slices with signal
        dropout (specific to dMRI; harmless for fMRI but rare).
    seed : int or None
        Seed for the per-call RNG. If None, a fresh RandomState is used.
    """

    def __init__(self,
                 p_case: float = 0.85,
                 case_scale_sigma: float = 1.0,
                 max_translation_vox: float = 4.0,
                 max_rotation_deg: float = 3.0,
                 max_through_plane_vox: float = 1.0,
                 n_ctrl_z: int = 4,
                 n_ctrl_t: int = 6,
                 jitter_sigma_vox: float = 0.25,
                 p_intensity: float = 0.7,
                 p_dropout: float = 0.15,
                 seed: int | None = None):
        self.p_case = p_case
        self.case_scale_sigma = case_scale_sigma
        self.max_translation_vox = max_translation_vox
        self.max_rotation_deg = max_rotation_deg
        self.max_through_plane_vox = max_through_plane_vox
        self.n_ctrl_z = n_ctrl_z
        self.n_ctrl_t = n_ctrl_t
        self.jitter_sigma_vox = jitter_sigma_vox
        self.p_intensity = p_intensity
        self.p_dropout = p_dropout
        self._seed = seed

    # -----------------------------------------------------------------------
    # Motion field sampling
    # -----------------------------------------------------------------------

    def _sample_motion_fields(self, D: int, T: int,
                              rng: np.random.Generator
                              ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample (Tx, Ty, Theta, Tz) of shape (D, T), each with realistic
        spatial/temporal smoothness and per-case amplitude.

        Returns
        -------
        Tx, Ty : (D, T) translations in voxels
        Theta  : (D, T) rotations in degrees
        Tz     : (D, T) through-plane shifts in voxels
        """
        # Per-case scale: half-normal + small floor. Most cases get small
        # motion, rare cases get large. The expected std of the resulting
        # motion field is roughly (scale * typical_amp), where typical_amp
        # is set per channel below.
        scale = abs(rng.normal(0, self.case_scale_sigma)) + 0.1

        def _build(amp_max: float, jitter_std: float, typical_frac: float = 0.25
                   ) -> np.ndarray:
            """
            Build a smooth motion field.

            typical_frac: typical std of the field is scale * typical_frac * amp_max.
            With case_scale_sigma=1.0 and typical_frac=0.25, most cases land
            around 0.1-0.4 * amp_max in std, occasional cases exceed amp_max
            and get clipped — exactly the heavy-tailed behaviour we want.
            """
            field = _smooth_random_field(D, T, self.n_ctrl_z, self.n_ctrl_t, rng)
            f = field * scale * typical_frac * amp_max
            f = f + rng.normal(0, jitter_std, size=f.shape).astype(np.float32)
            return np.clip(f, -amp_max, amp_max).astype(np.float32)

        Tx = _build(self.max_translation_vox, self.jitter_sigma_vox)
        Ty = _build(self.max_translation_vox, self.jitter_sigma_vox)
        Theta = _build(self.max_rotation_deg, self.jitter_sigma_vox * 0.5)
        Tz = _build(self.max_through_plane_vox, self.jitter_sigma_vox * 0.3)
        return Tx, Ty, Theta, Tz

    # -----------------------------------------------------------------------
    # Geometric transform
    # -----------------------------------------------------------------------

    def _apply_motion(self, vol: np.ndarray,
                      Tx: np.ndarray, Ty: np.ndarray, Theta: np.ndarray, Tz: np.ndarray
                      ) -> np.ndarray:
        """
        Apply slice-wise rigid motion + small through-plane shift.

        Order of operations (per timepoint):
          1) In-plane: rotate by Theta[d, t] then translate by (Tx, Ty)[d, t],
             slice by slice.
          2) Through-plane: shift along z by Tz[:, t].mean() — using the per-z
             mean keeps the warp coherent (avoids twisting the cord).
        """
        H, W, D, T = vol.shape
        out = np.empty_like(vol)

        for t in range(T):
            vol_t = vol[..., t]

            # 1) in-plane slicewise
            warped = np.empty_like(vol_t)
            for d in range(D):
                warped[:, :, d] = _per_slice_warp_2d(
                    vol_t[:, :, d],
                    tx=float(Tx[d, t]),
                    ty=float(Ty[d, t]),
                    theta_deg=float(Theta[d, t]),
                    order=1, mode="reflect",
                )

            # 2) through-plane: shift the whole 3D volume along z
            tz_mean = float(np.mean(Tz[:, t]))
            if abs(tz_mean) > 1e-3:
                yy, xx, zz = np.indices(warped.shape, dtype=np.float32)
                coords = np.stack([yy, xx, zz - tz_mean], axis=0)
                warped = map_coordinates(warped, coords, order=1,
                                         mode="reflect", cval=0.0)

            out[..., t] = warped

        return out

    # -----------------------------------------------------------------------
    # Intensity perturbations
    # -----------------------------------------------------------------------

    def _apply_intensity(self, vol: np.ndarray, mask: np.ndarray,
                         rng: np.random.Generator) -> np.ndarray:
        """
        Apply a stack of intensity perturbations to a 4D volume.

        Each perturbation is gated by its own probability so they compose
        (a sample may get noise + bias, just bias, or none).
        """
        out = vol.astype(np.float32, copy=True)
        H, W, D, T = out.shape

        # Reference scale for noise (intensity inside the mask)
        m = mask > 0
        if m.any():
            ref = float(out[..., 0][m].std() + 1e-6)
        else:
            ref = float(out[..., 0].std() + 1e-6)

        # --- Rician noise (acts on magnitude images; ~Gaussian for high SNR)
        if rng.random() < 0.7:
            sigma = ref * rng.uniform(0.005, 0.03)
            real = out + rng.normal(0, sigma, size=out.shape).astype(np.float32)
            imag = rng.normal(0, sigma, size=out.shape).astype(np.float32)
            out = np.sqrt(real * real + imag * imag).astype(np.float32)

        # --- Smooth bias field (one per timepoint, gentle)
        if rng.random() < 0.5:
            for t in range(T):
                # 8x8x4 random low-freq field upsampled to volume size
                low = rng.normal(0, 1, size=(8, 8, 4)).astype(np.float32)
                low = gaussian_filter(low, sigma=2.0)
                # Resample to (H, W, D)
                yy = np.linspace(0, low.shape[0] - 1, H)
                xx = np.linspace(0, low.shape[1] - 1, W)
                zz = np.linspace(0, low.shape[2] - 1, D)
                grid = np.stack(np.meshgrid(yy, xx, zz, indexing="ij"), axis=0)
                bias = map_coordinates(low, grid, order=3, mode="reflect")
                # Multiplicative bias, magnitude up to ±15%
                amp = rng.uniform(0.05, 0.15)
                bias_mult = 1.0 + amp * (bias / (np.abs(bias).max() + 1e-6))
                out[..., t] = out[..., t] * bias_mult.astype(np.float32)

        # --- Gamma (contrast)
        if rng.random() < 0.4:
            gamma = rng.uniform(0.8, 1.25)
            v = out
            vmin = v.min()
            vmax = v.max()
            if vmax > vmin:
                v_norm = (v - vmin) / (vmax - vmin)
                v_norm = np.power(np.clip(v_norm, 0, 1), gamma)
                out = (v_norm * (vmax - vmin) + vmin).astype(np.float32)

        return out

    def _apply_dropout(self, vol: np.ndarray,
                       rng: np.random.Generator) -> np.ndarray:
        """
        dMRI-style slice signal dropout: in a small fraction of timepoints,
        attenuate 1-2 random slices to simulate signal loss from spin-history
        effects during motion.
        """
        H, W, D, T = vol.shape
        out = vol.copy()
        for t in range(T):
            if rng.random() < self.p_dropout:
                n_drop = rng.integers(1, 3)  # 1 or 2 slices
                slices = rng.choice(D, size=n_drop, replace=False)
                for d in slices:
                    factor = rng.uniform(0.2, 0.6)  # attenuation, not zero
                    out[:, :, d, t] *= factor
        return out

    # -----------------------------------------------------------------------
    # Entry point
    # -----------------------------------------------------------------------

    def __call__(self, vol: np.ndarray, mask: np.ndarray | None = None
                 ) -> dict:
        """
        Augment a single 4D volume.

        Parameters
        ----------
        vol  : (H, W, D, T) float array — clean source volume
        mask : (H, W, D) array or None — used only to scale noise sensibly

        Returns
        -------
        dict with keys:
            "moving" : (H, W, D, T) float32 — augmented volume
            "Tx_gt"  : (D, T) float32 — applied x-translation per slice/time
            "Ty_gt"  : (D, T) float32 — applied y-translation per slice/time
            "applied_motion" : bool — whether motion was applied to this case
        """
        rng = np.random.default_rng(self._seed)

        if vol.ndim != 4:
            raise ValueError(f"Expected (H,W,D,T) 4D vol, got shape {vol.shape}")
        H, W, D, T = vol.shape
        if mask is None:
            mask = np.ones((H, W, D), dtype=np.float32)

        applied_motion = bool(rng.random() < self.p_case)

        if applied_motion:
            Tx, Ty, Theta, Tz = self._sample_motion_fields(D, T, rng)
            moving = self._apply_motion(vol, Tx, Ty, Theta, Tz)
        else:
            Tx = np.zeros((D, T), dtype=np.float32)
            Ty = np.zeros((D, T), dtype=np.float32)
            moving = vol.astype(np.float32, copy=True)

        # Intensity perturbations and dropout are applied regardless of
        # whether motion was applied — real images always have these.
        if rng.random() < self.p_intensity:
            moving = self._apply_intensity(moving, mask, rng)
        moving = self._apply_dropout(moving, rng)

        return {
            "moving": moving.astype(np.float32),
            "Tx_gt": Tx.astype(np.float32),
            "Ty_gt": Ty.astype(np.float32),
            "applied_motion": applied_motion,
        }


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Distributional sanity check across many synthetic cases
    rng_outer = np.random.default_rng(0)
    H, W, D, T = 64, 64, 14, 60   # realistic-ish for fMRI
    mask = np.zeros((H, W, D), dtype=np.float32)
    mask[20:44, 20:44, 2:12] = 1.0

    n_cases = 200
    motion_applied = []
    field_stds = []
    field_max_abs = []
    z_steps = []
    t_steps = []
    clip_rate = []  # fraction of (z,t) entries hitting the ±cap

    aug = MotionAugmenter()
    cap = aug.max_translation_vox

    for i in range(n_cases):
        vol = rng_outer.standard_normal((H, W, D, T)).astype(np.float32)
        # Use a fresh seed per call by setting the augmenter's seed
        aug._seed = int(rng_outer.integers(0, 2**31))
        out = aug(vol, mask)
        motion_applied.append(out["applied_motion"])
        if out["applied_motion"]:
            tx = out["Tx_gt"]
            field_stds.append(tx.std())
            field_max_abs.append(np.abs(tx).max())
            z_steps.append(np.abs(np.diff(tx, axis=0)).mean())
            t_steps.append(np.abs(np.diff(tx, axis=1)).mean())
            clip_rate.append(float((np.abs(tx) >= cap - 1e-3).mean()))

    print(f"Distributional check over {n_cases} cases (Tx only)")
    print(f"  motion-applied rate     : {np.mean(motion_applied):.2%}  (target ~85%)")
    print(f"  per-case std (vox)      : "
          f"median={np.median(field_stds):.2f}  p95={np.quantile(field_stds, 0.95):.2f}")
    print(f"  per-case max|Tx| (vox)  : "
          f"median={np.median(field_max_abs):.2f}  p95={np.quantile(field_max_abs, 0.95):.2f}  "
          f"cap={cap}")
    print(f"  cap-clip rate per case  : "
          f"mean={np.mean(clip_rate):.3%}  (low is good — heavy-tailed but not saturating)")
    print(f"  z-smoothness (mean |dTx/dz|, vox)  : "
          f"median={np.median(z_steps):.3f}")
    print(f"  t-smoothness (mean |dTx/dt|, vox)  : "
          f"median={np.median(t_steps):.3f}")
