"""
io/video_source.py
==================
Threaded video reader supporting webcam and file sources.
Runs a background thread to prevent frame drops during processing.

Usage:
    from io.video_source import VideoSource
    with VideoSource(config) as vs:
        while True:
            ok, frame, frame_num, ts = vs.read()
            if not ok:
                break
"""

import queue
import threading
import time

import cv2


class VideoSource:
    """
    Threaded video reader for webcam or video file input.

    A background thread continuously reads frames into a bounded queue,
    decoupling I/O from the main processing loop to prevent frame drops.

    Args:
        config: Dot-accessible config object from ConfigLoader.
                Reads: input.source, input.webcam_index,
                       memory.frame_buffer_size.

    Example:
        with VideoSource(config) as vs:
            props = vs.get_properties()
            ok, frame, fnum, ts = vs.read()
    """

    def __init__(self, config) -> None:
        self._config = config
        self._source = config.input.source
        self._webcam_index = config.input.webcam_index
        self._buffer_size = config.memory.frame_buffer_size

        self._cap: cv2.VideoCapture = None
        self._queue: queue.Queue = queue.Queue(maxsize=self._buffer_size)
        self._thread: threading.Thread = None
        self._stop_event = threading.Event()
        self._frame_number = 0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # Context manager                                                      #
    # ------------------------------------------------------------------ #

    def __enter__(self) -> "VideoSource":
        self._open()
        self._start_thread()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def read(self) -> tuple:
        """
        Read the next frame from the buffer queue.

        Blocks for up to 2 seconds waiting for a frame.

        Returns:
            tuple: (success, frame, frame_number, timestamp)
                   success (bool)       — False if source is exhausted or timed out
                   frame (np.ndarray)   — BGR frame
                   frame_number (int)   — 1-based frame index
                   timestamp (float)    — time.time() when frame was captured
        """
        try:
            item = self._queue.get(timeout=2.0)
            return item
        except queue.Empty:
            return False, None, -1, -1.0

    def get_properties(self) -> dict:
        """
        Return basic properties of the video source.

        Returns:
            dict: {width, height, fps, total_frames}
                  total_frames is -1 for webcam sources.
        """
        if self._cap is None:
            return {}
        return {
            "width":        int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height":       int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps":          self._cap.get(cv2.CAP_PROP_FPS),
            "total_frames": int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        }

    def release(self) -> None:
        """Stop the reader thread and release the video capture handle."""
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _open(self) -> None:
        """Open the video capture handle."""
        if self._source == "webcam":
            self._cap = cv2.VideoCapture(self._webcam_index)
        else:
            self._cap = cv2.VideoCapture(self._source)

        if not self._cap.isOpened():
            raise RuntimeError(
                f"[VideoSource] Cannot open source: '{self._source}'. "
                "Check config.yaml input.source."
            )

    def _start_thread(self) -> None:
        """Launch the background reader thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._reader_loop,
            name="VideoReaderThread",
            daemon=True,
        )
        self._thread.start()

    def _reader_loop(self) -> None:
        """
        Background thread: continuously reads frames into the queue.
        Stops when the source is exhausted or stop_event is set.
        """
        while not self._stop_event.is_set():
            ret, frame = self._cap.read()
            if not ret:
                # Signal end-of-stream to the consumer
                try:
                    self._queue.put((False, None, -1, -1.0), timeout=1.0)
                except queue.Full:
                    pass
                break

            with self._lock:
                self._frame_number += 1
                fnum = self._frame_number

            ts = time.time()

            try:
                self._queue.put((True, frame, fnum, ts), timeout=1.0)
            except queue.Full:
                # Drop oldest frame to stay current (real-time priority)
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._queue.put((True, frame, fnum, ts), timeout=0.5)
                except queue.Full:
                    pass