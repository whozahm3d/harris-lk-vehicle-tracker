"""
utils/health_check.py
=====================
Validates the environment before the tracking pipeline starts.
Checks OpenCV, video source, disk space, and config integrity.

Usage:
    from utils.health_check import HealthCheck
    HealthCheck().run(config, source="videos/test.mp4")
"""

import os
import shutil

import cv2


class HealthCheck:
    """
    Runs pre-flight checks before the tracking pipeline starts.

    Checks performed:
        - OpenCV importable and version printed
        - Video source openable (webcam index or file path)
        - Output directory writable
        - Sufficient disk space (warns if < 500 MB)
        - Config has no missing required keys

    Args:
        output_dir (str): Directory where results will be written.
                          Defaults to "results".

    Example:
        HealthCheck(output_dir="../results").run(cfg, source="webcam")
    """

    _REQUIRED_SECTIONS = (
        "input", "memory", "harris",
        "lucas_kanade", "tracking",
        "preprocessing", "output", "metrics", "cli",
    )
    _MIN_DISK_MB = 500

    def __init__(self, output_dir: str = "results") -> None:
        self._output_dir = output_dir
        self._passed: list[str] = []
        self._failed: list[str] = []

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def run(self, config, source: str = None) -> None:
        """
        Execute all checks and print a startup summary.

        Args:
            config: Dot-accessible config object from ConfigLoader.
            source (str, optional): Override source string. If None, reads
                                    from config.input.source.

        Raises:
            RuntimeError: If any critical check fails.
        """
        self._passed.clear()
        self._failed.clear()

        self._check_opencv()
        self._check_source(config, source)
        self._check_output_dir()
        self._check_disk_space()
        self._check_config_keys(config)

        self._print_summary()

        if self._failed:
            raise RuntimeError(
                "[HealthCheck] One or more critical checks failed. "
                "Fix the issues above before running the pipeline."
            )

    # ------------------------------------------------------------------ #
    # Individual checks                                                    #
    # ------------------------------------------------------------------ #

    def _check_opencv(self) -> None:
        """Verify OpenCV is importable and print its version."""
        try:
            version = cv2.__version__
            self._ok(f"OpenCV version: {version}")
        except Exception as e:
            self._fail(f"OpenCV not available: {e}")

    def _check_source(self, config, source: str = None) -> None:
        """Verify the video source can be opened."""
        src = source if source is not None else config.input.source
        if os.path.isfile(src):
            cap = cv2.VideoCapture(src)
            if cap.isOpened():
                self._ok(f"Video file accessible: {src}")
                cap.release()
            else:
                self._fail(f"File exists but cannot be opened by OpenCV: {src}")
        else:
            self._fail(f"Video file not found: {src}")

    def _check_output_dir(self) -> None:
        """Verify the output directory is writable."""
        try:
            os.makedirs(self._output_dir, exist_ok=True)
            test_file = os.path.join(self._output_dir, ".write_test")
            with open(test_file, "w") as f:
                f.write("ok")
            os.remove(test_file)
            self._ok(f"Output directory writable: {self._output_dir}")
        except Exception as e:
            self._fail(f"Output directory not writable: {e}")

    def _check_disk_space(self) -> None:
        """Warn if free disk space is below 500 MB."""
        try:
            usage = shutil.disk_usage(self._output_dir)
            free_mb = usage.free / (1024 ** 2)
            if free_mb < self._MIN_DISK_MB:
                self._fail(
                    f"Low disk space: {free_mb:.0f} MB free "
                    f"(minimum recommended: {self._MIN_DISK_MB} MB)"
                )
            else:
                self._ok(f"Disk space OK: {free_mb:.0f} MB free")
        except Exception as e:
            self._fail(f"Could not check disk space: {e}")

    def _check_config_keys(self, config) -> None:
        """Verify all required top-level config sections are present."""
        missing = [
            key for key in self._REQUIRED_SECTIONS
            if not hasattr(config, key)
        ]
        if missing:
            self._fail(f"Config missing sections: {missing}")
        else:
            self._ok("Config keys all present")

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _ok(self, message: str) -> None:
        self._passed.append(message)

    def _fail(self, message: str) -> None:
        self._failed.append(message)

    def _print_summary(self) -> None:
        print("\n" + "=" * 55)
        print("  ObjectTracker — Pre-flight Health Check")
        print("=" * 55)
        for msg in self._passed:
            print(f"  ✅  {msg}")
        for msg in self._failed:
            print(f"  ❌  {msg}")
        print("=" * 55 + "\n")