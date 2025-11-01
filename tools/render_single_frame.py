from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from road_visibility.config import PipelineConfig
from road_visibility.visibility import RoadVisibilityEstimator


def draw_overlay(frame: np.ndarray, estimator: RoadVisibilityEstimator) -> np.ndarray:
    overlay = frame.copy()
    estimate = estimator.estimate(frame)

    boundary = estimator.boundary_detector.detect(
        frame,
        estimate.vanish_point_row,
    )

    roi_mask: Optional[np.ndarray] = None
    if estimate.roi_mask is not None:
        roi_mask = estimate.roi_mask
    elif boundary.roi_mask is not None:
        roi_mask = boundary.roi_mask

    if roi_mask is not None:
        tint = np.zeros_like(frame)
        tint[:, :, 2] = 255
        overlay[roi_mask > 0] = overlay[roi_mask > 0] * 0.35 + tint[roi_mask > 0] * 0.65

    if boundary.left is not None:
        overlay = cv2.line(
            overlay,
            (int(boundary.left.x_at(frame.shape[0] - 1)), frame.shape[0] - 1),
            (
                int(boundary.left.x_at(int(max(0, estimate.vanish_point_row)))),
                int(max(0, estimate.vanish_point_row)),
            ),
            (255, 0, 0),
            2,
        )
    if boundary.right is not None:
        overlay = cv2.line(
            overlay,
            (int(boundary.right.x_at(frame.shape[0] - 1)), frame.shape[0] - 1),
            (
                int(boundary.right.x_at(int(max(0, estimate.vanish_point_row)))),
                int(max(0, estimate.vanish_point_row)),
            ),
            (0, 0, 255),
            2,
        )

    for seg in estimate.lane_segments:
        contour = seg.contour.astype(np.int32)
        if contour.shape[0] >= 2:
            cv2.polylines(overlay, [contour], isClosed=False, color=(0, 255, 0), thickness=2)

    vp = (int(round(estimate.vanish_point_col)), int(round(estimate.vanish_point_row)))
    cv2.drawMarker(
        overlay,
        vp,
        (0, 0, 255),
        markerType=cv2.MARKER_CROSS,
        markerSize=18,
        thickness=2,
    )
    cv2.putText(
        overlay,
        f"vp=({estimate.vanish_point_col:.1f},{estimate.vanish_point_row:.1f}) "
        f"dashes={len(estimate.lane_segments)}",
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return overlay


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render lane/vanish detection overlay for a single frame.")
    parser.add_argument("--image", required=True, help="Path to the input image.")
    parser.add_argument("--output", help="Path to save the overlay. Defaults to <image>_overlay.png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_path = Path(args.image)
    output_path = Path(args.output) if args.output else image_path.with_name(f"{image_path.stem}_overlay.png")
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")

    config = PipelineConfig()
    estimator = RoadVisibilityEstimator(config=config)
    frame_proc = estimator._resize_frame(frame)
    estimator.initialize(frame_proc)
    overlay = draw_overlay(frame_proc, estimator)
    if overlay.shape[:2] != frame.shape[:2]:
        overlay = cv2.resize(overlay, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), overlay)
    print(f"[info] overlay saved to {output_path}")


if __name__ == "__main__":
    main()
