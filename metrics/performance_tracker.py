"""
metrics/performance_tracker.py
===============================
Tracks per-frame performance metrics: FPS, active point count,
active object count, and bbox stability. Exports to JSON and CSV.

Usage:
    from metrics.performance_tracker import PerformanceTracker
    pt = PerformanceTracker(config, logger)
    pt.update(frame_id=1, fps=28.4, point_count=143, object_count=3, bbox_list=[(x,y,w,h)])
    pt.export_json(path)
    pt.export_csv(path)
    print(pt.summary())
"""

import csv
import json
import time

import numpy as np


class PerformanceTracker:
    """
    Records per-frame metrics and computes run-level statistics.

    Tracked per frame:
        - frame_id        : frame index
        - timestamp       : wall-clock time (seconds since epoch)
        - fps             : frames per second at that frame
        - point_count     : total active Harris corners across all trackers
        - object_count    : number of active trackers
        - bbox_stability  : mean bbox diagonal length (proxy for stability)

    Args:
        config: Dot-accessible config object from ConfigLoader.
                Reads: metrics.log_interval
        logger: AppLogger instance.

    Example:
        pt = PerformanceTracker(cfg, logger)
        pt.update(1, 29.1, 150, 2, [(100,80,60,40)])
        pt.export_csv("results/metrics/metrics.csv")
    """

    def __init__(self, config, logger) -> None:
        self._log_interval = config.metrics.log_interval
        self._logger       = logger
        self._records: list[dict] = []
        self._start_time   = time.time()

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def update(
        self,
        frame_id: int,
        fps: float,
        point_count: int,
        object_count: int,
        bbox_list: list,
    ) -> None:
        """
        Record metrics for a single frame.

        Args:
            frame_id     (int):   1-based frame index.
            fps          (float): Measured FPS at this frame.
            point_count  (int):   Total tracked Harris corners across all objects.
            object_count (int):   Number of active trackers.
            bbox_list    (list):  List of (x, y, w, h) for all active trackers.
        """
        stability = self._mean_diagonal(bbox_list)

        record = {
            "frame_id":      frame_id,
            "timestamp":     round(time.time() - self._start_time, 3),
            "fps":           round(fps, 2),
            "point_count":   point_count,
            "object_count":  object_count,
            "bbox_stability": round(stability, 2),
        }
        self._records.append(record)

        if frame_id % self._log_interval == 0:
            self._logger.info(
                f"Frame {frame_id} | FPS: {fps:.1f} | "
                f"Objects: {object_count} | Points: {point_count}"
            )

    def export_json(self, path: str) -> None:
        """
        Export all recorded metrics to a JSON file.

        Args:
            path (str): Full path to output .json file.
        """
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._records, f, indent=2)
        self._logger.info(f"Metrics exported to JSON: {path}")

    def export_csv(self, path: str) -> None:
        """
        Export all recorded metrics to a CSV file.

        Args:
            path (str): Full path to output .csv file.
        """
        if not self._records:
            self._logger.warning("No metrics to export.")
            return

        fieldnames = list(self._records[0].keys())
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self._records)
        self._logger.info(f"Metrics exported to CSV: {path}")

    def summary(self) -> str:
        """
        Return a human-readable summary string of the full run.

        Returns:
            str: Multi-line summary with avg/min/max for key metrics.
        """
        if not self._records:
            return "No metrics recorded."

        fps_vals    = [r["fps"]          for r in self._records]
        pt_vals     = [r["point_count"]  for r in self._records]
        obj_vals    = [r["object_count"] for r in self._records]
        total_time  = self._records[-1]["timestamp"]

        lines = [
            "=" * 48,
            "  ObjectTracker — Run Summary",
            "=" * 48,
            f"  Total frames    : {len(self._records)}",
            f"  Total time      : {total_time:.1f}s",
            f"  FPS  avg/min/max: {np.mean(fps_vals):.1f} / {np.min(fps_vals):.1f} / {np.max(fps_vals):.1f}",
            f"  Points avg      : {np.mean(pt_vals):.0f}",
            f"  Objects avg     : {np.mean(obj_vals):.1f}",
            "=" * 48,
        ]
        return "\n".join(lines)

    def get_dataframe(self):
        """
        Return metrics as a pandas DataFrame.

        Returns:
            pandas.DataFrame: One row per frame.
        """
        import pandas as pd
        return pd.DataFrame(self._records)

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _mean_diagonal(bbox_list: list) -> float:
        """
        Compute mean bounding box diagonal length across all active trackers.

        Args:
            bbox_list (list): List of (x, y, w, h) tuples.

        Returns:
            float: Mean diagonal in pixels, or 0.0 if list is empty.
        """
        if not bbox_list:
            return 0.0
        diags = [np.sqrt(w ** 2 + h ** 2) for _, _, w, h in bbox_list]
        return float(np.mean(diags))