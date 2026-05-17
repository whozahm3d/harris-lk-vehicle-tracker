"""
visualization/plotter.py
========================
Generates and saves all result plots as .png files at end of run.
Accepts a pandas DataFrame of per-frame metrics from PerformanceTracker.

Usage:
    from visualization.plotter import Plotter
    p = Plotter(config, logger)
    p.generate_all(df, trail_data, frame_shape, plots_dir)
"""

import os

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for headless/Jupyter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class Plotter:
    """
    Generates and saves all result plots from a metrics DataFrame.

    Plots produced:
        - fps_chart.png          : FPS over frames (line chart)
        - point_survival.png     : Active point count over frames (line chart)
        - object_count.png       : Active object count over frames (bar/line)
        - tracking_trails.png    : All centroid trails on a blank canvas

    Args:
        config: Dot-accessible config object from ConfigLoader.
                Reads: output.save_plots
        logger: AppLogger instance.

    Example:
        p = Plotter(cfg, logger)
        p.generate_all(df, trail_data, (480, 640), "results/plots")
    """

    _FIG_SIZE = (10, 4)   # width x height in inches for all line charts
    _DPI      = 120

    def __init__(self, config, logger) -> None:
        self._save_plots = config.output.save_plots
        self._logger     = logger

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def generate_all(
        self,
        df: pd.DataFrame,
        trail_data: dict,
        frame_shape: tuple,
        plots_dir: str,
    ) -> None:
        """
        Generate and save all four plots.

        Args:
            df          (pd.DataFrame): Metrics DataFrame from PerformanceTracker.
            trail_data  (dict):         {object_id: [(cx,cy), ...]} centroid histories.
            frame_shape (tuple):        (height, width) of the processed frame.
            plots_dir   (str):          Directory to save .png files into.
        """
        if not self._save_plots:
            self._logger.info("Plotting disabled in config (output.save_plots=false).")
            return

        self.plot_fps(df,            os.path.join(plots_dir, "fps_chart.png"))
        self.plot_point_survival(df, os.path.join(plots_dir, "point_survival.png"))
        self.plot_object_count(df,   os.path.join(plots_dir, "object_count.png"))
        self.plot_tracking_trails(trail_data, frame_shape,
                                  os.path.join(plots_dir, "tracking_trails.png"))

        self._logger.info(f"All plots saved to: {plots_dir}")

    def plot_fps(self, df: pd.DataFrame, path: str) -> None:
        """
        Save a line chart of FPS over frames.

        Args:
            df   (pd.DataFrame): Must contain columns 'frame_id' and 'fps'.
            path (str):          Output path for fps_chart.png.
        """
        fig, ax = plt.subplots(figsize=self._FIG_SIZE, dpi=self._DPI)
        ax.plot(df["frame_id"], df["fps"], color="#2196F3", linewidth=1.2, label="FPS")
        ax.axhline(df["fps"].mean(), color="#FF5722", linestyle="--",
                   linewidth=1.0, label=f"Avg: {df['fps'].mean():.1f}")
        ax.set_title("FPS Over Frames", fontsize=13, fontweight="bold")
        ax.set_xlabel("Frame")
        ax.set_ylabel("FPS")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        self._save(fig, path)

    def plot_point_survival(self, df: pd.DataFrame, path: str) -> None:
        """
        Save a line chart of total active Harris point count over frames.

        Args:
            df   (pd.DataFrame): Must contain 'frame_id' and 'point_count'.
            path (str):          Output path for point_survival.png.
        """
        fig, ax = plt.subplots(figsize=self._FIG_SIZE, dpi=self._DPI)
        ax.plot(df["frame_id"], df["point_count"], color="#4CAF50",
                linewidth=1.2, label="Active Points")
        ax.fill_between(df["frame_id"], df["point_count"], alpha=0.15, color="#4CAF50")
        ax.set_title("Active Harris Points Over Frames", fontsize=13, fontweight="bold")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Point Count")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        self._save(fig, path)

    def plot_object_count(self, df: pd.DataFrame, path: str) -> None:
        """
        Save a combined bar + line chart of active object count over frames.

        Args:
            df   (pd.DataFrame): Must contain 'frame_id' and 'object_count'.
            path (str):          Output path for object_count.png.
        """
        fig, ax = plt.subplots(figsize=self._FIG_SIZE, dpi=self._DPI)
        ax.bar(df["frame_id"], df["object_count"], color="#9C27B0",
               alpha=0.4, label="Object Count")
        ax.plot(df["frame_id"], df["object_count"].rolling(15, min_periods=1).mean(),
                color="#9C27B0", linewidth=1.5, label="15-frame rolling avg")
        ax.set_title("Active Object Count Over Frames", fontsize=13, fontweight="bold")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Object Count")
        ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        self._save(fig, path)

    def plot_tracking_trails(
        self,
        trail_data: dict,
        frame_shape: tuple,
        path: str,
    ) -> None:
        """
        Draw all centroid trails on a blank dark canvas and save as PNG.

        Args:
            trail_data  (dict):  {object_id (int): [(cx, cy), ...]} centroid lists.
            frame_shape (tuple): (height, width) of processing frame — sets canvas size.
            path        (str):   Output path for tracking_trails.png.
        """
        h, w  = frame_shape[:2]
        ratio = h / w
        fig_w = 10
        fig_h = max(3.0, fig_w * ratio)

        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=self._DPI)
        ax.set_facecolor("#0d0d0d")
        fig.patch.set_facecolor("#0d0d0d")

        cmap = plt.get_cmap("hsv")

        for obj_id, trail in trail_data.items():
            if len(trail) < 2:
                continue
            pts   = np.array(trail, dtype=np.float32)
            color = cmap((obj_id * 37 % 180) / 180.0)

            # Draw segments with fading alpha
            n = len(pts)
            for i in range(1, n):
                alpha = 0.2 + 0.8 * (i / n)
                ax.plot(
                    [pts[i - 1, 0], pts[i, 0]],
                    [pts[i - 1, 1], pts[i, 1]],
                    color=color, alpha=alpha, linewidth=1.5,
                )

            # Mark start and end
            ax.scatter(pts[0, 0],  pts[0, 1],  c=[color], s=30, zorder=5, marker="o")
            ax.scatter(pts[-1, 0], pts[-1, 1], c=[color], s=50, zorder=5, marker="*")
            ax.text(pts[-1, 0] + 3, pts[-1, 1] - 3, f"ID {obj_id}",
                    color=color, fontsize=7, va="top")

        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)   # flip Y to match image coordinate system
        ax.set_title("Centroid Tracking Trails", color="white", fontsize=13, fontweight="bold")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444444")

        fig.tight_layout()
        self._save(fig, path)

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _save(self, fig: plt.Figure, path: str) -> None:
        """
        Save a matplotlib figure to disk and close it to free memory.

        Args:
            fig  (plt.Figure): Figure to save.
            path (str):        Full output file path (.png).
        """
        try:
            fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
            self._logger.info(f"Plot saved: {path}")
        except Exception as e:
            self._logger.error(f"Failed to save plot {path}: {e}")
        finally:
            plt.close(fig)