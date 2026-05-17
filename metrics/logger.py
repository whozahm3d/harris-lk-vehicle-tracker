"""
metrics/logger.py
=================
Singleton logger used by every module in the project.
Writes INFO+ to console and DEBUG+ to results/logs/run.log.

Usage:
    from metrics.logger import AppLogger
    logger = AppLogger.get_logger()
    logger.info("Tracker started.")
"""

import logging
import os
from typing import Optional


class AppLogger:
    """
    Singleton application logger with console and file handlers.

    All modules must use this logger instead of bare print() statements.
    The file handler writes to the path supplied on first initialisation;
    subsequent calls to get_logger() return the same instance.

    Args:
        log_path (str, optional): Full path to the log file.
                                  Only used on the very first call.

    Example:
        logger = AppLogger.get_logger("results/run_001/logs/run.log")
        logger.info("Pipeline started.")
        logger.debug("Frame 42 processed.")
    """

    _instance: Optional[logging.Logger] = None

    @classmethod
    def get_logger(cls, log_path: Optional[str] = None) -> logging.Logger:
        """
        Return the singleton Logger, initialising it if this is the first call.

        Args:
            log_path (str, optional): Absolute or relative path to the log file.
                                      Required on the first call; ignored after.

        Returns:
            logging.Logger: Configured logger instance.

        Raises:
            ValueError: If called for the first time without a log_path.
        """
        if cls._instance is not None:
            return cls._instance

        if log_path is None:
            raise ValueError(
                "[AppLogger] log_path must be provided on the first call to get_logger()."
            )

        # Ensure the log directory exists
        log_dir = os.path.dirname(log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        logger = logging.getLogger("ObjectTracker")
        logger.setLevel(logging.DEBUG)           # capture everything at root level

        # ── Guard against duplicate handlers (Jupyter kernel restarts) ────
        # logging.getLogger() returns the same internal logger object even
        # after a module reload, so handlers can accumulate across restarts.
        if logger.handlers:
            logger.handlers.clear()

        # ── Console handler (INFO and above) ──────────────────────────────
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_fmt = logging.Formatter(
            fmt="[%(levelname)s] %(message)s"
        )
        console_handler.setFormatter(console_fmt)

        # ── File handler (DEBUG and above) ────────────────────────────────
        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_fmt = logging.Formatter(
            fmt="%(asctime)s  [%(levelname)-8s]  %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_fmt)

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

        # Prevent log messages from propagating to the root logger,
        # which can cause duplicate output in some Jupyter environments.
        logger.propagate = False

        cls._instance = logger
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """
        Reset the singleton (useful for unit tests or re-initialisation).
        Removes all handlers from the existing logger before clearing it.

        Always call this in your Jupyter cache-clear cell before reloading
        modules, to ensure the next get_logger() call re-attaches handlers
        cleanly:

            AppLogger.reset()
            for mod in list(sys.modules.keys()):
                if any(x in mod for x in ["core","utils","metrics","visualization"]):
                    del sys.modules[mod]
        """
        if cls._instance is not None:
            for handler in cls._instance.handlers[:]:
                handler.close()
                cls._instance.removeHandler(handler)
            cls._instance = None