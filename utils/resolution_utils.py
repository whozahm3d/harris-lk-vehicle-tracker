"""
utils/resolution_utils.py
=========================
Module-level functions for frame resolution management.
Handles downscaling frames for processing and upscaling points
back to display resolution.

Usage:
    from utils.resolution_utils import downscale_frame, upscale_points, get_scale_factor
"""

import cv2
import numpy as np


def get_scale_factor(original_wh: tuple, target_wh: tuple) -> tuple:
    """
    Compute the (sx, sy) scale factors between two resolutions.

    Args:
        original_wh (tuple): Original (width, height).
        target_wh   (tuple): Target (width, height).

    Returns:
        tuple: (sx, sy) where sx = target_w / original_w, sy = target_h / original_h.

    Example:
        sx, sy = get_scale_factor((1280, 720), (640, 480))
        # sx = 0.5, sy = 0.666...
    """
    orig_w, orig_h = original_wh
    tgt_w, tgt_h = target_wh
    return (tgt_w / orig_w, tgt_h / orig_h)


def downscale_frame(frame: np.ndarray, target_wh: tuple) -> np.ndarray:
    """
    Resize a frame to the target (width, height) for processing.

    Args:
        frame     (np.ndarray): Input BGR or grayscale frame.
        target_wh (tuple):      Target resolution as (width, height).

    Returns:
        np.ndarray: Resized frame.

    Example:
        small = downscale_frame(frame, (640, 480))
    """
    return cv2.resize(frame, target_wh, interpolation=cv2.INTER_LINEAR)


def upscale_points(points: np.ndarray, scale_factor: tuple) -> np.ndarray:
    """
    Map tracked points from process resolution back to display resolution.

    Args:
        points       (np.ndarray): Points array of shape (N, 1, 2) or (N, 2),
                                   dtype float32.
        scale_factor (tuple):      (sx, sy) from get_scale_factor().

    Returns:
        np.ndarray: Scaled points with the same shape as input.

    Example:
        sx, sy = get_scale_factor((640, 480), (1280, 720))
        display_pts = upscale_points(process_pts, (sx, sy))
    """
    if points is None or len(points) == 0:
        return points

    sx, sy = scale_factor
    scaled = points.copy().astype(np.float32)

    if scaled.ndim == 3:          # shape (N, 1, 2)
        scaled[:, :, 0] *= sx
        scaled[:, :, 1] *= sy
    elif scaled.ndim == 2:        # shape (N, 2)
        scaled[:, 0] *= sx
        scaled[:, 1] *= sy

    return scaled


def upscale_bbox(bbox: tuple, scale_factor: tuple) -> tuple:
    """
    Scale a bounding box from process resolution to display resolution.

    Args:
        bbox         (tuple): (x, y, w, h) in process resolution.
        scale_factor (tuple): (sx, sy) from get_scale_factor().

    Returns:
        tuple: (x, y, w, h) in display resolution, all as integers.

    Example:
        display_bbox = upscale_bbox((100, 80, 60, 40), (2.0, 1.5))
        # (200, 120, 120, 60)
    """
    sx, sy = scale_factor
    x, y, w, h = bbox
    return (int(x * sx), int(y * sy), int(w * sx), int(h * sy))