"""
core/harris_detector.py
=======================
Harris Corner Detection with optional CLAHE preprocessing and ROI masking.
Used to detect stable, distinctive feature points within a bounding box.

Usage:
    from core.harris_detector import HarrisDetector
    detector = HarrisDetector(config)
    corners = detector.detect(gray_frame, mask=roi_mask)
"""

import cv2
import numpy as np


class HarrisDetector:
    """
    Detects Harris corners within a grayscale frame or masked region.

    Wraps cv2.goodFeaturesToTrack() with Harris mode enabled.
    Optionally applies CLAHE for illumination normalization before detection.

    Args:
        config: Dot-accessible config object from ConfigLoader.
                Reads: harris.*, preprocessing.*

    Example:
        detector = HarrisDetector(config)
        corners = detector.detect(gray_frame)                  # full frame
        corners = detector.detect(gray_frame, mask=bbox_mask)  # within bbox
    """

    def __init__(self, config) -> None:
        self._max_corners    = config.harris.max_corners
        self._quality_level  = config.harris.quality_level
        self._min_distance   = config.harris.min_distance
        self._block_size     = config.harris.block_size
        self._k              = config.harris.k

        self._clahe_enable   = config.preprocessing.clahe_enable
        self._clahe          = self._build_clahe(config) if self._clahe_enable else None

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def detect(self, gray_frame: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        """
        Detect Harris corners in a grayscale frame.

        Args:
            gray_frame (np.ndarray): Grayscale input frame (uint8).
            mask (np.ndarray, optional): Binary mask (uint8). Corners are only
                                         detected where mask == 255.

        Returns:
            np.ndarray: Corner points of shape (N, 1, 2), dtype float32.
                        Returns empty array of shape (0, 1, 2) if none found.
        """
        processed = self._preprocess(gray_frame)

        corners = cv2.goodFeaturesToTrack(
            processed,
            maxCorners=self._max_corners,
            qualityLevel=self._quality_level,
            minDistance=self._min_distance,
            mask=mask,
            blockSize=self._block_size,
            useHarrisDetector=True,
            k=self._k,
        )

        if corners is None:
            return np.empty((0, 1, 2), dtype=np.float32)

        return corners.astype(np.float32)

    def create_bbox_mask(self, frame_shape: tuple, bbox: tuple) -> np.ndarray:
        """
        Create a binary mask for a bounding box region.

        Args:
            frame_shape (tuple): (height, width) of the frame.
            bbox (tuple):        (x, y, w, h) bounding box.

        Returns:
            np.ndarray: uint8 mask where the bbox region is 255, rest is 0.
        """
        h, w = frame_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        x, y, bw, bh = bbox
        # Clamp to frame boundaries
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(w, x + bw)
        y2 = min(h, y + bh)
        mask[y1:y2, x1:x2] = 255
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        return mask

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _preprocess(self, gray_frame: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE to normalize illumination before detection.

        Args:
            gray_frame (np.ndarray): Input grayscale frame.

        Returns:
            np.ndarray: CLAHE-enhanced frame, or original if CLAHE disabled.
        """
        if self._clahe_enable and self._clahe is not None:
            return self._clahe.apply(gray_frame)
        return gray_frame

    @staticmethod
    def _build_clahe(config) -> cv2.CLAHE:
        """
        Build a CLAHE object from config parameters.

        Args:
            config: Dot-accessible config object.

        Returns:
            cv2.CLAHE: Configured CLAHE instance.
        """
        tile = tuple(config.preprocessing.clahe_tile_size)
        return cv2.createCLAHE(
            clipLimit=config.preprocessing.clahe_clip_limit,
            tileGridSize=tile,
        )