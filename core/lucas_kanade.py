"""
core/lucas_kanade.py
====================
Lucas-Kanade Optical Flow tracker with forward-backward error checking.
"""

import cv2
import numpy as np


class LKTracker:
    def __init__(self, config) -> None:
        self._win_size        = tuple(config.lucas_kanade.win_size)
        self._max_level       = config.lucas_kanade.max_level
        self._max_iter        = config.lucas_kanade.max_iter
        self._epsilon         = config.lucas_kanade.epsilon
        self._min_eig_thresh  = config.lucas_kanade.min_eig_threshold
        self._fb_error_thresh = config.lucas_kanade.fb_error_threshold

        self._clahe_enable = config.preprocessing.clahe_enable
        self._clahe        = self._build_clahe(config) if self._clahe_enable else None

        self._criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            self._max_iter,
            self._epsilon,
        )

    def track(self, prev_gray, curr_gray, prev_points):
        empty = np.empty((0, 1, 2), dtype=np.float32)

        if prev_points is None or len(prev_points) == 0:
            return empty, empty, np.array([], dtype=bool)

        prev_e = self._preprocess(prev_gray)
        curr_e = self._preprocess(curr_gray)

        # Forward pass
        new_pts, st_fwd, _ = cv2.calcOpticalFlowPyrLK(
            prev_e, curr_e, prev_points, None,
            winSize=self._win_size,
            maxLevel=self._max_level,
            criteria=self._criteria,
            flags=cv2.OPTFLOW_LK_GET_MIN_EIGENVALS,
            minEigThreshold=self._min_eig_thresh,
        )

        if new_pts is None or st_fwd is None:
            return empty, empty, np.array([], dtype=bool)

        # Backward pass
        back_pts, st_bwd, _ = cv2.calcOpticalFlowPyrLK(
            curr_e, prev_e, new_pts, None,
            winSize=self._win_size,
            maxLevel=self._max_level,
            criteria=self._criteria,
        )

        # Forward-backward error
        if back_pts is not None:
            fb_error = np.linalg.norm(
                prev_points.reshape(-1, 2) - back_pts.reshape(-1, 2),
                axis=1,
            )
            fb_mask = fb_error < self._fb_error_thresh
        else:
            fb_mask = np.zeros(len(prev_points), dtype=bool)

        fwd_mask   = st_fwd.ravel() == 1
        final_mask = fwd_mask & fb_mask

        if not np.any(final_mask):
            return empty, empty, final_mask

        return new_pts[final_mask], prev_points[final_mask], final_mask

    def _preprocess(self, gray_frame):
        if self._clahe_enable and self._clahe is not None:
            return self._clahe.apply(gray_frame)
        return gray_frame

    @staticmethod
    def _build_clahe(config):
        tile = tuple(config.preprocessing.clahe_tile_size)
        return cv2.createCLAHE(
            clipLimit=config.preprocessing.clahe_clip_limit,
            tileGridSize=tile,
        )