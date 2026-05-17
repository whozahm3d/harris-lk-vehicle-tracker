"""
main.py
=======
Single-object Harris + LK tracker.
Reads start frame and bbox from config.yaml (set once in Phase 2).
Run from phase5_integration.ipynb Cell 2.
"""

import sys
import ctypes
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    pass
import cv2
import numpy as np
import os

from utils.config_loader import ConfigLoader
from utils.folder_manager import FolderManager
from metrics.logger import AppLogger
from core.object_tracker import ObjectTracker
from visualization.display import DisplayRenderer


def main(config_path="config.yaml"):
    # ── Load config & setup ───────────────────────────────────────────
    config     = ConfigLoader(config_path).config
    input_name = os.path.splitext(os.path.basename(config.input.source))[0]
    fm         = FolderManager(video_name=input_name, base_dir="results")
    log_path   = os.path.join(fm.get_path("logs"), "run.log")
    logger     = AppLogger.get_logger(log_path)

    logger.info("Starting Harris + LK single-object tracker.")

    # ── Open video ────────────────────────────────────────────────────
    cap = cv2.VideoCapture(config.input.source)
    if not cap.isOpened():
        logger.error(f"Cannot open video: {config.input.source}")
        sys.exit(1)

    fps    = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0 or fps > 240:
        fps = 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info(f"Video: {width}x{height} @ {fps:.1f} fps")

    # ── Load frame index and bbox from config (set once in Phase 2) ──
    chosen_frame_idx = config.input.start_frame
    bbox             = tuple(config.input.bbox)

    cap.set(cv2.CAP_PROP_POS_FRAMES, chosen_frame_idx)
    ret, chosen_frame = cap.read()
    if not ret:
        logger.error(f"Could not read frame {chosen_frame_idx} from video.")
        sys.exit(1)

    logger.info(f"Frame {chosen_frame_idx} selected as start frame.")
    logger.info(f"Selected bbox (full res): {bbox}")

    # ── Initialize tracker on chosen frame ───────────────────────────
    gray_first = cv2.cvtColor(chosen_frame, cv2.COLOR_BGR2GRAY)
    tracker    = ObjectTracker(object_id=1, config=config, logger=logger)
    tracker.initialize(chosen_frame, gray_first, bbox)

    # ── Video writer ──────────────────────────────────────────────────
    writer      = None
    output_path = None
    if config.output.save_video:
        input_name  = os.path.splitext(os.path.basename(config.input.source))[0]
        output_path = os.path.join(fm.get_path("videos"), f"{input_name}_tracked.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"avc1")  # H.264 codec
        writer      = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        logger.info(f"Output: {output_path}")

    # ── Live display window setup ─────────────────────────────────────
    DISP_W, DISP_H = 1280, 720
    WIN = "Harris + LK Tracker — press Q to quit"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, DISP_W, DISP_H)
    cv2.moveWindow(WIN, 0, 0)

    renderer  = DisplayRenderer(config)
    prev_gray = gray_first
    frame_idx = chosen_frame_idx

    # ── Annotate and display start frame ─────────────────────────────
    init_out = renderer.draw_single(chosen_frame, tracker.get_display_data(), frame_idx)
    if writer:
        writer.write(init_out)
    display = cv2.resize(init_out, (DISP_W, DISP_H), interpolation=cv2.INTER_LINEAR)
    cv2.imshow(WIN, display)
    cv2.waitKey(1)

    # ── Main tracking loop ────────────────────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        tracker.update(prev_gray, curr_gray, frame)

        out = renderer.draw_single(frame, tracker.get_display_data(), frame_idx)

        if writer:
            writer.write(out)

        if not tracker.active:
            logger.info("Tracker lost object. Stopping.")
            break

        # ── Live display ──────────────────────────────────────────────
        display = cv2.resize(out, (DISP_W, DISP_H), interpolation=cv2.INTER_LINEAR)
        cv2.imshow(WIN, display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            logger.info("User quit early via Q key.")
            break

        if frame_idx % config.metrics.log_interval == 0:
            data = tracker.get_display_data()
            logger.info(f"Frame {frame_idx} | Points: {data['point_count']}")

        prev_gray = curr_gray

    # ── Cleanup ───────────────────────────────────────────────────────
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    logger.info(f"Done. Processed {frame_idx - chosen_frame_idx} frames.")
    if output_path:
        logger.info(f"Output saved to: {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    main(args.config)