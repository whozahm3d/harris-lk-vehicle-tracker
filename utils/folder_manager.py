"""
utils/folder_manager.py
=======================
Creates and manages the structured results output folder for every run.

Usage:
    from utils.folder_manager import FolderManager
    fm = FolderManager(video_name="test_video")
    print(fm.get_path("videos"))   # results/test_video_20240101_120000/videos
"""

import json
import os
import shutil
from datetime import datetime
from typing import Dict


# Valid output categories
_CATEGORIES = ("videos", "plots", "metrics", "params", "logs")


class FolderManager:
    """
    Creates a timestamped results folder and all required subdirectories
    at the start of every run.

    Folder structure created:
        results/{video_name}_{YYYYMMDD_HHMMSS}/
            videos/
            plots/
            metrics/
            params/
            logs/

    Args:
        video_name (str): Base name used in the folder (e.g. "test_video" or "webcam").
        base_dir (str): Root directory for all results. Defaults to "results".

    Example:
        fm = FolderManager("test_video")
        log_dir = fm.get_path("logs")       # pass to AppLogger
        fm.save_config_snapshot("config.yaml")
        fm.save_run_info({"source": "webcam", "fps": 30})
    """

    def __init__(self, video_name: str, base_dir: str = "results") -> None:
        self._video_name = video_name
        self._timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._run_name = f"{video_name}_{self._timestamp}"
        self._run_dir = os.path.join(base_dir, self._run_name)
        self._paths: Dict[str, str] = {}
        self._create_structure()

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def get_path(self, category: str) -> str:
        """
        Return the absolute path for a given output category.

        Args:
            category (str): One of 'videos', 'plots', 'metrics', 'params', 'logs'.

        Returns:
            str: Full path to the requested subdirectory.

        Raises:
            KeyError: If an invalid category is requested.
        """
        if category not in self._paths:
            raise KeyError(
                f"[FolderManager] Unknown category '{category}'. "
                f"Valid options: {_CATEGORIES}"
            )
        return self._paths[category]

    @property
    def paths(self) -> dict:
        """Return all category paths as a plain dict (for phase4 compatibility)."""
        return self._paths
    def get_run_dir(self) -> str:
        """
        Return the root directory for this run.

        Returns:
            str: Full path to results/{video_name}_{timestamp}/
        """
        return self._run_dir

    def save_config_snapshot(self, config_path: str) -> None:
        """
        Copy config.yaml into the params/ subdirectory for reproducibility.

        Args:
            config_path (str): Path to the source config.yaml file.

        Raises:
            FileNotFoundError: If config_path does not exist.
        """
        if not os.path.isfile(config_path):
            raise FileNotFoundError(
                f"[FolderManager] config.yaml not found at: {config_path}"
            )
        dest = os.path.join(self._paths["params"], "config_snapshot.yaml")
        shutil.copy2(config_path, dest)

    def save_run_info(self, info_dict: dict) -> None:
        """
        Save run metadata as run_info.json in the params/ subdirectory.

        Args:
            info_dict (dict): Metadata to save. Recommended keys:
                              source, resolution, date, opencv_version.
        """
        info_dict["run_name"] = self._run_name
        info_dict["timestamp"] = self._timestamp
        dest = os.path.join(self._paths["params"], "run_info.json")
        with open(dest, "w", encoding="utf-8") as f:
            json.dump(info_dict, f, indent=4)

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _create_structure(self) -> None:
        """Create the run directory and all required subdirectories."""
        for category in _CATEGORIES:
            path = os.path.join(self._run_dir, category)
            os.makedirs(path, exist_ok=True)
            self._paths[category] = path