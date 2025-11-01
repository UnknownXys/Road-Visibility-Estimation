from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from road_visibility.config import PipelineConfig
from road_visibility.detectors.lane_markings import LaneDashDetector


@dataclass
class FittedLine:
    slope: float  # x = slope * y + intercept
    intercept: float

    def x_at(self, y: float) -> float:
        return self.slope * y + self.intercept


@dataclass
class DashSegment:
    start: Tuple[int, int]
    end: Tuple[int, int]
    length: float


def _build_candidate_mask(gray: np.ndarray, config: PipelineConfig, roi_mask: np.ndarray) -> np.ndarray:
    kernel_size = max(3, config.lane_tophat_kernel | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    blended = cv2.addWeighted(tophat, config.lane_tophat_weight, gray, 1.0 - config.lane_tophat_weight, 0.0)

    block_size = max(3, config.lane_adaptive_threshold_block | 1)
    adaptive = cv2.adaptiveThreshold(
        blended,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        config.lane_adaptive_threshold_c,
    )
    _, otsu = cv2.threshold(blended, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    combined = cv2.bitwise_or(adaptive, otsu)
    combined = cv2.bitwise_and(combined, roi_mask)

    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (3, max(3, config.lane_vertical_kernel | 1)),
    )
    processed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, vertical_kernel, iterations=1)
    processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, vertical_kernel, iterations=1)
    processed = cv2.dilate(processed, vertical_kernel, iterations=1)
    return processed


def _weighted_fit(points: Sequence[Tuple[float, float]], weights: Sequence[float]) -> Optional[FittedLine]:
    if len(points) < 4:
        return None
    ys = np.array([p[1] for p in points], dtype=np.float64)
    xs = np.array([p[0] for p in points], dtype=np.float64)
    ws = np.sqrt(np.maximum(np.array(weights, dtype=np.float64), 1e-6))
    slope, intercept = np.polyfit(ys, xs, 1, w=ws)
    residuals = np.abs(slope * ys + intercept - xs)
    median = float(np.median(residuals))
    if median > 1e-3:
        keep = residuals <= median * 3.0
        if np.count_nonzero(keep) >= 4:
            slope, intercept = np.polyfit(ys[keep], xs[keep], 1, w=ws[keep])
    return FittedLine(float(slope), float(intercept))


def _collect_boundary_lines(
    lines: np.ndarray,
    width: int,
    approx_vanish_row: float,
    config: PipelineConfig,
) -> Tuple[Optional[FittedLine], Optional[FittedLine]]:
    left_points: List[Tuple[Tuple[float, float], float]] = []
    right_points: List[Tuple[Tuple[float, float], float]] = []

    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        length = math.hypot(x2 - x1, y2 - y1)
        if length < config.boundary_min_line_length:
            continue
        if max(y1, y2) < approx_vanish_row - config.boundary_roi_expand_px * 0.5:
            continue
        if y1 < y2:
            top = (x1, y1)
            bottom = (x2, y2)
        else:
            top = (x2, y2)
            bottom = (x1, y1)
        dy = bottom[1] - top[1]
        if abs(dy) < 1e-3:
            continue
        dx = bottom[0] - top[0]
        angle = abs(math.degrees(math.atan2(dx, dy)))
        weight = length * (1.0 + max(0.0, (bottom[1] - approx_vanish_row) / max(config.boundary_weight_bottom_bias, 1.0)))
        ratio = bottom[0] / max(1.0, width)

        if ratio <= config.boundary_min_bottom_ratio and angle >= config.boundary_left_min_vertical_angle:
            left_points.append((top, weight))
            left_points.append((bottom, weight))
        elif ratio >= config.boundary_min_bottom_ratio and angle <= config.boundary_right_max_vertical_angle:
            right_points.append((top, weight))
            right_points.append((bottom, weight))

    left_line = _weighted_fit([p for p, _ in left_points], [w for _, w in left_points])
    right_line = _weighted_fit([p for p, _ in right_points], [w for _, w in right_points])
    return left_line, right_line


def _sample_line_intensity(
    image: np.ndarray,
    start: Tuple[float, float],
    end: Tuple[float, float],
    width: int,
) -> float:
    mask = np.zeros_like(image, dtype=np.uint8)
    p0 = (int(round(start[0])), int(round(start[1])))
    p1 = (int(round(end[0])), int(round(end[1])))
    cv2.line(mask, p0, p1, 255, thickness=max(1, width))
    values = image[mask == 255]
    if values.size == 0:
        return 0.0
    return float(np.mean(values))


def render_detection(image_path: Path, output_path: Path, config: PipelineConfig) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, config.boundary_canny_low, config.boundary_canny_high, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=45, minLineLength=50, maxLineGap=20)
    lines_array = lines if lines is not None else np.empty((0, 1, 4), dtype=np.float32)
    approx_vanish_row = height * config.roi_top_ratio
    left_line, right_line = _collect_boundary_lines(lines_array, width, approx_vanish_row, config)

    if left_line is None or right_line is None:
        raise RuntimeError("Failed to fit boundary lines.")

    if abs(left_line.slope - right_line.slope) > 1e-3:
        vanish_y = (right_line.intercept - left_line.intercept) / (left_line.slope - right_line.slope)
        vanish_x = left_line.x_at(vanish_y)
    else:
        vanish_y = approx_vanish_row
        vanish_x = (left_line.x_at(vanish_y) + right_line.x_at(vanish_y)) * 0.5

    vanish_y = float(max(-config.vanish_point_offset_limit, min(height * 1.2, vanish_y)))
    vanish_x = float(max(-width * 0.5, min(width * 1.5, vanish_x)))

    roi_bottom = height - 1
    top_row = max(0.0, vanish_y)
    tail_row = max(0.0, top_row - config.boundary_roi_tail_px)

    roi_points: List[Tuple[int, int]] = [
        (int(round(left_line.x_at(roi_bottom))), roi_bottom),
        (int(round(right_line.x_at(roi_bottom))), roi_bottom),
        (int(round(right_line.x_at(top_row))), int(round(top_row))),
        (int(round(left_line.x_at(top_row))), int(round(top_row))),
    ]
    if tail_row < top_row:
        roi_points.append((int(round(right_line.x_at(tail_row))), int(round(tail_row))))
        roi_points.append((int(round(left_line.x_at(tail_row))), int(round(tail_row))))

    roi_polygon = np.array(roi_points, dtype=np.int32)
    roi_mask = np.zeros_like(gray, dtype=np.uint8)
    cv2.fillConvexPoly(roi_mask, roi_polygon, 255)

    dash_detector = LaneDashDetector(config)
    dash_segments = [
        DashSegment(start=tuple(seg.contour[0]), end=tuple(seg.contour[1]), length=seg.length_px)
        for seg in dash_detector.detect(
            image,
            vanish_point_row=vanish_y,
            vanish_point_col=vanish_x,
            roi_mask=roi_mask,
        )
    ]

    overlay = image.copy()
    cv2.polylines(overlay, [roi_polygon], isClosed=True, color=(0, 165, 255), thickness=2)

    bottom = int(roi_bottom)
    top = int(max(0, min(height - 1, vanish_y)))
    left_start = (int(round(left_line.x_at(bottom))), bottom)
    left_end = (int(round(left_line.x_at(top))), top)
    right_start = (int(round(right_line.x_at(bottom))), bottom)
    right_end = (int(round(right_line.x_at(top))), top)
    cv2.line(overlay, left_start, left_end, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.line(overlay, right_start, right_end, (0, 0, 255), 2, cv2.LINE_AA)

    for segment in dash_segments:
        cv2.line(overlay, segment.start, segment.end, (0, 255, 0), 2, cv2.LINE_AA)

    vp_point = (int(round(vanish_x)), int(round(vanish_y)))
    cv2.drawMarker(
        overlay,
        vp_point,
        (0, 0, 255),
        markerType=cv2.MARKER_CROSS,
        markerSize=20,
        thickness=2,
    )
    cv2.putText(
        overlay,
        f"vp=({vanish_x:.1f},{vanish_y:.1f})  dashes={len(dash_segments)}",
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    cv2.imwrite(str(output_path), overlay)


def main() -> None:
    config = PipelineConfig()
    image_path = Path("clear_refs/G1523K315354_clear.jpg")
    output_path = Path("tuned_debug/G1523K315354_detection_overlay.jpg")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    render_detection(image_path, output_path, config)
    print(f"[info] saved overlay to {output_path}")


if __name__ == "__main__":
    main()
