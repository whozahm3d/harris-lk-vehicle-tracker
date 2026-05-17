"""
core/object_tracker.py
======================
Tracks a single user-defined object using Harris corners + LK optical flow.
Median displacement shift for bbox updates.
No Kalman. No scale estimation. Matches assignment spec exactly.
"""

import cv2
import numpy as np
from collections import deque

from core.harris_detector import HarrisDetector
from core.lucas_kanade import LKTracker


class ObjectTracker:
    def __init__(self, object_id, config, logger) -> None:
        self.object_id       = object_id
        self.bbox            = None
        self.centroid        = None
        self.points          = None
        self.trail           = deque(maxlen=config.memory.max_trail_length)
        self.active          = True
        self.inactive_frames = 0

        self._cfg             = config
        self._logger          = logger
        self._harris          = HarrisDetector(config)
        self._lk              = LKTracker(config)

        self._min_pts         = config.tracking.min_points_per_object
        self._redetect_thresh = config.harris.redetect_threshold
        self._max_inactive    = config.memory.max_inactive_frames
        self._smoothing       = config.tracking.bbox_smoothing
        self._redetect_every  = config.tracking.redetect_interval
        self._frame_count     = 0
        self._initial_pts     = 1
        self._velocity        = (0, 0)   # (dx, dy) running estimate

    def initialize(self, frame, gray_frame, bbox):
        self.bbox     = bbox
        mask          = self._harris.create_bbox_mask(gray_frame.shape, bbox)
        self.points   = self._harris.detect(gray_frame, mask=mask)
        self._initial_pts = max(len(self.points), 1)
        self.centroid = self._compute_centroid(bbox)
        self.trail.append(self.centroid)
        self._logger.debug(
            f"[Tracker {self.object_id}] Initialized with {len(self.points)} points."
        )

    def update(self, prev_gray, curr_gray, curr_frame):
        if not self.active:
            return False

        self._frame_count += 1

        # LK tracking
        good_new, good_prev, _ = self._lk.track(prev_gray, curr_gray, self.points)

        if len(good_new) >= self._min_pts:
            # Filter out stationary background points when object is moving
            speed = np.sqrt(self._velocity[0]**2 + self._velocity[1]**2)
            if speed > 3.0:
                displacements = good_new[:, 0, :] - good_prev[:, 0, :]
                disp_mag      = np.sqrt(displacements[:, 0]**2 + displacements[:, 1]**2)
                motion_mask   = disp_mag > speed * 0.3
                if motion_mask.sum() >= self._min_pts:
                    good_new  = good_new[motion_mask]
                    good_prev = good_prev[motion_mask]

            new_bbox  = self._median_shift_bbox(good_prev, good_new, self.bbox)
            old_bbox  = self.bbox
            self.bbox = self._smooth_bbox(self.bbox, new_bbox)
            self.points       = good_new
            self.inactive_frames = 0

            # Update velocity from good tracking
            dx = self.bbox[0] - old_bbox[0]
            dy = self.bbox[1] - old_bbox[1]
            alpha = 0.6
            self._velocity = (
                alpha * dx + (1 - alpha) * self._velocity[0],
                alpha * dy + (1 - alpha) * self._velocity[1],
            )

            # Redetect if point survival is low
            survival = len(good_new) / self._initial_pts
            if survival < self._redetect_thresh:
                self._redetect(curr_frame, curr_gray)

        else:
            if len(good_new) > 0:
                new_bbox     = self._median_shift_bbox(good_prev, good_new, self.bbox)
                measured_dx  = new_bbox[0] - self.bbox[0]
                measured_dy  = new_bbox[1] - self.bbox[1]

                # Blend measured displacement with velocity prediction
                # Weight velocity more when fewer points survive
                point_ratio = len(good_new) / self._min_pts
                w = max(0.0, min(1.0, point_ratio - 1.0))
                dx = w * measured_dx + (1 - w) * self._velocity[0]
                dy = w * measured_dy + (1 - w) * self._velocity[1]

                x, y, bw, bh   = self.bbox
                predicted_bbox = (int(round(x + dx)), int(round(y + dy)), bw, bh)
                self.bbox      = self._smooth_bbox(self.bbox, predicted_bbox)
                self.points    = good_new
            else:
                # Zero points — use pure velocity to keep bbox moving
                x, y, bw, bh = self.bbox
                self.bbox = (
                    int(round(x + self._velocity[0])),
                    int(round(y + self._velocity[1])),
                    bw, bh
                )

            # Redetect inside the shifted bbox position
            self._redetect(curr_frame, curr_gray)

            if len(self.points) < self._min_pts:
                self.inactive_frames += 1
                self._logger.debug(
                    f"[Tracker {self.object_id}] Low points. "
                    f"Inactive: {self.inactive_frames}"
                )

        if self.inactive_frames > self._max_inactive:
            self.active = False
            self._logger.info(f"[Tracker {self.object_id}] Lost — deactivated.")
            return False

        if self._bbox_out_of_frame(self.bbox, curr_gray.shape):
            self.active = False
            self._logger.info(f"[Tracker {self.object_id}] Left frame — deactivated.")
            return False

        self.centroid = self._compute_centroid(self.bbox)
        self.trail.append(self.centroid)
        return True

    def get_display_data(self):
        return {
            "id":          self.object_id,
            "bbox":        self.bbox,
            "centroid":    self.centroid,
            "points":      self.points,
            "trail":       list(self.trail),
            "point_count": len(self.points) if self.points is not None else 0,
            "active":      self.active,
        }

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    def _redetect(self, frame, gray_frame):
        if self.bbox is None:
            return
        mask    = self._harris.create_bbox_mask(gray_frame.shape, self.bbox)
        new_pts = self._harris.detect(gray_frame, mask=mask)
        if len(new_pts) > 0:
            self.points       = new_pts
            self._initial_pts = max(len(new_pts), 1)
            self._logger.debug(
                f"[Tracker {self.object_id}] Redetected {len(new_pts)} points."
            )

    @staticmethod
    def _median_shift_bbox(prev_pts, next_pts, bbox):
        displacements = next_pts[:, 0, :] - prev_pts[:, 0, :]
        dx, dy = np.median(displacements, axis=0)
        x, y, w, h = bbox
        return (int(round(x + dx)), int(round(y + dy)), w, h)

    def _smooth_bbox(self, old_bbox, new_bbox):
        if old_bbox is None:
            return new_bbox
        a = self._smoothing
        return tuple(int(a * o + (1 - a) * n) for o, n in zip(old_bbox, new_bbox))

    @staticmethod
    def _compute_centroid(bbox):
        x, y, w, h = bbox
        return (int(x + w / 2), int(y + h / 2))

    @staticmethod
    def _bbox_out_of_frame(bbox, frame_shape):
        if bbox is None:
            return False
        h, w = frame_shape[:2]
        x, y, bw, bh = bbox
        ix1 = max(0, x);  iy1 = max(0, y)
        ix2 = min(w, x + bw);  iy2 = min(h, y + bh)
        visible = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        total   = bw * bh
        return (visible / total) < 0.05 if total > 0 else True