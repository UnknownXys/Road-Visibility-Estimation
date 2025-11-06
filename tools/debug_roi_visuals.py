from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from road_visibility.config import PipelineConfig
from road_visibility.detectors.lane_boundaries import LaneBoundaryDetector


def main() -> None:
    parser = argparse.ArgumentParser(description="Dump ROI debugging visuals for a single frame.")
    parser.add_argument("--image", required=True, help="Path to the input image (clear frame).")
    parser.add_argument(
        "--output-dir",
        help="Output directory (defaults to <image>_debug alongside the input).",
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise FileNotFoundError(image_path)

    config = PipelineConfig()
    height, width = frame.shape[:2]
    approx_row = height * config.roi_top_ratio

    detector = LaneBoundaryDetector(config)
    detection = detector.detect(frame, approx_row)

    out_dir = Path(args.output_dir) if args.output_dir else image_path.with_suffix("").with_name(f"{image_path.stem}_debug")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ROI overlay
    overlay = frame.copy()
    if detection.roi_mask is not None:
        tint = np.zeros_like(frame)
        tint[:, :, 2] = 255
        overlay[detection.roi_mask > 0] = overlay[detection.roi_mask > 0] * 0.35 + tint[detection.roi_mask > 0] * 0.65
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
    cv2.imwrite(str(out_dir / "roi_overlay.png"), overlay)

    # LSD lines
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_ADV)
    lines = lsd.detect(blurred)[0]

    lsd_canvas = frame.copy()
    if lines is not None:
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            cv2.line(
                lsd_canvas,
                (int(round(x1)), int(round(y1))),
                (int(round(x2)), int(round(y2))),
                (0, 255, 255),
                1,
            )
    cv2.imwrite(str(out_dir / "lsd_all.png"), lsd_canvas)

    # Candidate lines used for boundary fitting
    candidate_canvas = frame.copy()
    if lines is not None:
        scale_x = max(width / max(config.frame_reference_width, 1e-6), 1e-6)
        scale_y = max(height / max(config.frame_reference_height, 1e-6), 1e-6)
        scale_len = (scale_x + scale_y) * 0.5
        min_len = config.boundary_min_line_length * scale_len
        row_limit = approx_row - config.boundary_roi_expand_px * scale_y * 0.5
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            length = float(np.hypot(x2 - x1, y2 - y1))
            if length < min_len or max(y1, y2) < row_limit:
                continue
            cv2.line(
                candidate_canvas,
                (int(round(x1)), int(round(y1))),
                (int(round(x2)), int(round(y2))),
                (0, 255, 0),
                2,
            )
    cv2.imwrite(str(out_dir / "lsd_candidates.png"), candidate_canvas)

    print("left:", detection.left)
    print("right:", detection.right)
    print("vanish:", detection.vanish_point)


if __name__ == "__main__":
    main()
