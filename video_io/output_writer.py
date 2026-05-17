"""
io/output_writer.py
===================
Saves the annotated tracking output as an mp4 video file.

Usage:
    from io.output_writer import OutputWriter
    with OutputWriter(path, resolution=(1280,720), fps=30.0) as ow:
        ow.write(annotated_frame)
"""

import cv2
import numpy as np


class OutputWriter:
    """
    Writes annotated frames to an mp4 video file using the mp4v codec.

    Args:
        output_path (str):    Full path to the output .mp4 file.
        resolution  (tuple):  (width, height) of output frames.
        fps         (float):  Frames per second for the output video.

    Example:
        with OutputWriter("results/videos/output_tracked.mp4", (1280,720), 30.0) as ow:
            for frame in annotated_frames:
                ow.write(frame)
    """

    def __init__(self, output_path: str, resolution: tuple, fps: float) -> None:
        self._output_path = output_path
        self._resolution = resolution        # (width, height)
        self._fps = fps
        self._writer: cv2.VideoWriter = None
        self._frame_count = 0

    # ------------------------------------------------------------------ #
    # Context manager                                                      #
    # ------------------------------------------------------------------ #

    def __enter__(self) -> "OutputWriter":
        self._open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def write(self, frame: np.ndarray) -> None:
        """
        Write a single annotated frame to the output video.

        Resizes frame if it does not match the configured resolution.

        Args:
            frame (np.ndarray): BGR frame to write.
        """
        if self._writer is None:
            return

        h, w = frame.shape[:2]
        target_w, target_h = self._resolution

        if (w, h) != (target_w, target_h):
            frame = cv2.resize(frame, self._resolution, interpolation=cv2.INTER_LINEAR)

        try:
            self._writer.write(frame)
            self._frame_count += 1
        except Exception:
            # Never crash the tracking loop due to a write failure
            pass

    def release(self) -> None:
        """Flush and close the video writer."""
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    @property
    def frame_count(self) -> int:
        """Return the number of frames written so far."""
        return self._frame_count

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _open(self) -> None:
        """Initialise the cv2.VideoWriter with mp4v codec."""
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(
            self._output_path,
            fourcc,
            self._fps,
            self._resolution,
        )
        if not self._writer.isOpened():
            raise RuntimeError(
                f"[OutputWriter] Failed to open VideoWriter at: {self._output_path}"
            )