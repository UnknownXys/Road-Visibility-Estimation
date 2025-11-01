from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from road_visibility.config import PipelineConfig
from road_visibility.detectors.lane_boundaries import LaneBoundaryDetector


def main() -> None:
    config = PipelineConfig()
    image_path = Path("tuned_debug/G1523K319513_segment_clear/clear_reference.png")
    if not image_path.exists():
        raise FileNotFoundError(image_path)
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise RuntimeError(f"Failed to read {image_path}")

    height, width = frame.shape[:2]
    approx_row = height * config.roi_top_ratio

    detector = LaneBoundaryDetector(config)
    detection = detector.detect(frame, approx_row)

    overlay = frame.copy()
    if detection.roi_mask is not None:
        mask = detection.roi_mask
        tint = np.zeros_like(frame)
        tint[:, :, 2] = 255
        overlay[mask > 0] = overlay[mask > 0] * 0.3 + tint[mask > 0] * 0.7
    if detection.left is not None:
        overlay = cv2.line(
            overlay,
            (int(detection.left.x_at(height - 1)), height - 1),
            (int(detection.left.x_at(int(max(0, approx_row)))), int(max(0, approx_row))),
            (255, 0, 0),
            2,
        )
    if detection.right is not None:
        overlay = cv2.line(
            overlay,
            (int(detection.right.x_at(height - 1)), height - 1),
            (int(detection.right.x_at(int(max(0, approx_row)))), int(max(0, approx_row))),
            (0, 0, 255),
            2,
        )
    out_dir = image_path.parent
    cv2.imwrite(str(out_dir / "clear_reference_with_roi.png"), overlay)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_ADV)
    lines = lsd.detect(blurred)[0]

    all_lsd = frame.copy()
    if lines is not None:
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            cv2.line(all_lsd, (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2))), (0, 255, 255), 1)
    cv2.imwrite(str(out_dir / "clear_reference_all_lsd.png"), all_lsd)

    candidate_canvas = frame.copy()
    if lines is not None:
        scale_x = max(width / max(config.frame_reference_width, 1e-6), 1e-6)
        scale_y = max(height / max(config.frame_reference_height, 1e-6), 1e-6)
        scale_len = (scale_x + scale_y) * 0.5
        min_len = config.boundary_min_line_length * scale_len
        row_limit = approx_row - config.boundary_roi_expand_px * scale_y * 0.5
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            length = float(np.hypot(x2 - x1, y2 - y1))
            if length < min_len:
                continue
            if max(y1, y2) < row_limit:
                continue
            cv2.line(
                candidate_canvas,
                (int(round(x1)), int(round(y1))),
                (int(round(x2)), int(round(y2))),
                (0, 255, 0),
                2,
            )
    cv2.imwrite(str(out_dir / "clear_reference_candidate_lines.png"), candidate_canvas)

    print("left:", detection.left)
    print("right:", detection.right)
    print("vanish:", detection.vanish_point)


if __name__ == "__main__":
    main()
