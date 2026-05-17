"""
visualization/display.py
========================
Renders tracking output onto video frames for a single tracked object.

Usage:
    from visualization.display import DisplayRenderer
    renderer = DisplayRenderer(config)
    out_frame = renderer.draw_single(frame, tracker.get_display_data(), frame_idx)
"""

import cv2


class DisplayRenderer:
    """
    Draws bounding box, corner points, trail, and HUD onto a frame.

    Args:
        config: Dot-accessible config object from ConfigLoader.
    """

    def __init__(self, config) -> None:
        self._show_trails = config.output.show_trails
        self._trail_length = config.output.trail_length

    def draw_single(self, frame, data, frame_idx):
        """Draw tracking output for a single object."""
        out = frame.copy()
        bbox     = data.get("bbox")
        points   = data.get("points")
        trail    = data.get("trail", [])
        n_points = data.get("point_count", 0)

        # Trail
        if self._show_trails:
            for i in range(1, len(trail)):
                alpha = i / len(trail)
                colour = tuple(int(c * alpha) for c in (180, 180, 60))
                cv2.line(out, trail[i-1], trail[i], colour,
                         max(1, int(alpha * 3)), cv2.LINE_AA)

        # Bounding box
        if bbox is not None:
            x, y, w, h = bbox
            cv2.rectangle(out, (x, y), (x+w, y+h), (0, 220, 0), 2, cv2.LINE_AA)
            tick = min(15, w // 4, h // 4)
            for px, py, sx, sy in [
                (x,     y,     1,  1),
                (x + w, y,    -1,  1),
                (x,     y + h, 1, -1),
                (x + w, y + h,-1, -1),
            ]:
                cv2.line(out, (px, py), (px + sx * tick, py),    (0, 220, 0), 3, cv2.LINE_AA)
                cv2.line(out, (px, py), (px, py + sy * tick),    (0, 220, 0), 3, cv2.LINE_AA)
            # Centroid
            cx, cy = int(x + w / 2), int(y + h / 2)
            cv2.circle(out, (cx, cy), 5, (255, 0, 255), -1, cv2.LINE_AA)

        # Corner points
        if points is not None:
            for pt in points:
                cv2.circle(out, (int(pt[0, 0]), int(pt[0, 1])),
                           3, (0, 255, 255), -1, cv2.LINE_AA)

        # HUD
        lines = [f"Frame:  {frame_idx:04d}", f"Points: {n_points}", "Status: TRACKING"]
        x0, y0, pad, lh = 10, 10, 6, 18
        bh = lh * len(lines) + pad * 2
        cv2.rectangle(out, (x0, y0), (x0 + 185, y0 + bh), (20, 20, 20), -1)
        cv2.rectangle(out, (x0, y0), (x0 + 185, y0 + bh), (80, 80, 80),  1)
        for i, line in enumerate(lines):
            cv2.putText(out, line, (x0 + pad, y0 + pad + (i + 1) * lh - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (240, 240, 240), 1, cv2.LINE_AA)
        return out