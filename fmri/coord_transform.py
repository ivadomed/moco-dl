"""
Utilities for transferring a rigid (rotation + translation) affine transform,
predicted in one image canvas's normalized coordinate space, into a DIFFERENT
canvas's coordinate space -- specifically: a transform predicted on a spinal-cord
crop needs to be re-expressed so it can be applied directly to the raw, full-FOV
image the crop was taken from.

Why this isn't just "reuse the same theta": F.affine_grid's theta is defined
relative to the canvas it's used with -- rotation happens about that canvas's
center, and translation is expressed as a fraction of that canvas's own
half-width/half-height. The crop and the raw image are different-sized canvases
with different centers, so the same numeric theta means a physically different
transform on each. This module converts theta into a canvas-independent PIXEL
affine (rotate by the same angle about the crop's own center, then translate by
a fixed number of pixels), then re-expresses that pixel affine as the correct
theta for the raw canvas.

All conversions use the continuous approximation of affine_grid's align_corners=False
coordinate convention (norm = 2*pixel/size - 1), consistent with the rest of this
project's rigid_model.py.
"""

import numpy as np


def pixel_affine_from_theta(theta: np.ndarray, canvas_shape) -> tuple:
    """
    Convert a (2, 3) normalized affine_grid matrix into an equivalent PIXEL-space
    affine: pixel_in = A @ pixel_out + b_px, where pixel_in/pixel_out are (x, y)
    pixel coordinates within a canvas of the given (H, W) shape.
    """
    H, W = canvas_shape
    A = theta[:, :2]               # (2, 2) linear part -- same rotation matrix regardless of canvas
    theta_trans = theta[:, 2]      # (2,) translation part, in (x, y) order

    center_px = np.array([W / 2.0, H / 2.0])   # (x, y)
    S = np.array([W / 2.0, H / 2.0])           # per-axis normalized->pixel scale

    b_px = (np.eye(2) - A) @ center_px + S * theta_trans
    return A, b_px


def theta_from_pixel_affine(A: np.ndarray, b_px: np.ndarray, canvas_shape) -> np.ndarray:
    """Inverse of pixel_affine_from_theta: build the (2, 3) theta for a canvas of given shape."""
    H, W = canvas_shape
    center_px = np.array([W / 2.0, H / 2.0])
    S = np.array([W / 2.0, H / 2.0])

    theta_trans = (b_px - (np.eye(2) - A) @ center_px) / S
    return np.concatenate([A, theta_trans[:, None]], axis=1)  # (2, 3)


def transfer_theta_to_raw(theta_model: np.ndarray, model_shape, crop_center_in_raw_px, raw_shape) -> np.ndarray:
    """
    Transfer a rigid theta predicted on the model-space crop canvas into the
    equivalent theta for the raw full-FOV canvas.

    theta_model: (2, 3) predicted affine, as used with the model-space canvas
                 (e.g. the padded/cropped-to-TARGET_SHAPE crop image).
    model_shape: (H, W) of the model-space canvas.
    crop_center_in_raw_px: (x, y) location, in RAW image pixel coordinates, of
                 the crop's own center -- this is also where the model-space
                 canvas's center physically sits, since to_model_space pads/crops
                 symmetrically around the crop's content.
    raw_shape: (H, W) of the raw full-FOV canvas.

    Returns theta_raw: (2, 3) affine to use with F.affine_grid on the raw canvas.
    """
    A, b_px_model = pixel_affine_from_theta(theta_model, model_shape)

    model_center_px = np.array([model_shape[1] / 2.0, model_shape[0] / 2.0])  # (x, y)
    crop_center_in_raw_px = np.asarray(crop_center_in_raw_px, dtype=np.float64)

    # Constant pixel offset between model-space coordinates and raw-space
    # coordinates: model-space's own center maps to the crop's center in raw space.
    offset = crop_center_in_raw_px - model_center_px

    # pixel_in_raw = A @ pixel_out_raw + b_px_raw, derived by substituting
    # pixel_model = pixel_raw - offset into pixel_in_model = A@pixel_out_model + b_px_model
    b_px_raw = (np.eye(2) - A) @ offset + b_px_model

    return theta_from_pixel_affine(A, b_px_raw, raw_shape)
