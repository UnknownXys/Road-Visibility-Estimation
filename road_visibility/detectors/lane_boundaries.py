from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
from math import atan2, degrees, hypot

from ..config import PipelineConfig
from ..utils import polygon_from_vanish_point


@dataclass
class BoundaryLine:
    vx: float
    vy: float
    x0: float
    y0: float

    def x_at(self, y: float) -> Optional[float]:
        if abs(self.vy) < 1e-6:
            return None
        t = (y - self.y0) / self.vy
        return float(self.x0 + t * self.vx)

    def angle_deg(self) -> float:
        return float(abs(np.degrees(np.arctan2(self.vy, self.vx))))

    def as_points(self, y0: float, y1: float) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        x_start = self.x_at(y0)
        x_end = self.x_at(y1)
        if x_start is None or x_end is None:
            return (0, int(y0)), (0, int(y1))
        return (int(round(x_start)), int(round(y0))), (int(round(x_end)), int(round(y1)))


@dataclass
class BoundaryDetection:
    left: Optional[BoundaryLine]
    right: Optional[BoundaryLine]
    vanish_point: Optional[Tuple[float, float]]
    roi_mask: Optional[np.ndarray]


class LaneBoundaryDetector:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    def detect(self, frame: np.ndarray, approx_vanish_row: float) -> BoundaryDetection:
        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        detector = cv2.createLineSegmentDetector(cv2.LSD_REFINE_ADV)
        lines = detector.detect(blurred)[0]
        if lines is None:
            return BoundaryDetection(None, None, None, None)

        scale_x = max(width / max(self.config.frame_reference_width, 1e-6), 1e-6)
        scale_y = max(height / max(self.config.frame_reference_height, 1e-6), 1e-6)
        scale_len = (scale_x + scale_y) * 0.5

        min_line_length = self.config.boundary_min_line_length * scale_len
        expand_px = self.config.boundary_roi_expand_px * scale_y
        tail_px = self.config.boundary_roi_tail_px * scale_y
        weight_bias = max(self.config.boundary_weight_bottom_bias * scale_y, 1.0)

        candidates = self._collect_candidates(
            lines,
            approx_vanish_row,
            width,
            min_line_length,
            approx_vanish_row - expand_px * 0.5,
        )
        left_points, left_slope = self._cluster_candidates(candidates, width, weight_bias, side="left")
        right_points, right_slope = self._cluster_candidates(candidates, width, weight_bias, side="right")

        left_line = self._fit_boundary(left_points, left_slope)
        right_line = self._fit_boundary(right_points, right_slope)

        if (left_line is None) ^ (right_line is None):
            relaxed_angle_epsilon = 20.0  # disable angle filter in fallback to recover the missing side
            if left_line is None and left_points:
                left_line = self._fit_boundary(left_points, left_slope, angle_epsilon=relaxed_angle_epsilon)
            if right_line is None and right_points:
                right_line = self._fit_boundary(right_points, right_slope, angle_epsilon=relaxed_angle_epsilon)

        vanish: Optional[Tuple[float, float]] = None
        roi_mask: Optional[np.ndarray] = None

        if left_line and right_line:
            intersection = self._intersect_lines(left_line, right_line)
            if intersection is not None and 0.0 <= intersection[1] <= height * 1.5:
                vanish = intersection
            else:
                left_at_row = left_line.x_at(approx_vanish_row)
                right_at_row = right_line.x_at(approx_vanish_row)
                if left_at_row is not None and right_at_row is not None:
                    vanish_x = (left_at_row + right_at_row) * 0.5
                    vanish = (vanish_x, float(approx_vanish_row))
            roi_mask = self._build_roi_mask(
                height,
                width,
                left_line,
                right_line,
                approx_vanish_row,
                int(round(expand_px)),
                int(round(tail_px)),
            )

        return BoundaryDetection(left_line, right_line, vanish, roi_mask)

    def _collect_candidates(
        self,
        lines: np.ndarray,
        approx_vanish_row: float,
        width: int,
        min_line_length: float,
        row_limit: float,
    ) -> List[Tuple[Tuple[float, float], Tuple[float, float], float, float]]:
        candidates: List[Tuple[Tuple[float, float], Tuple[float, float], float, float]] = []
        for line in lines:
            x1, y1, x2, y2 = map(float, line[0])
            length = float(hypot(x2 - x1, y2 - y1))
            if length < min_line_length:
                continue
            if max(y1, y2) < row_limit:
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
            angle = abs(degrees(atan2(dx, dy)))
            candidates.append((top, bottom, length, angle))
        return candidates

    def _cluster_candidates(
        self,
        candidates: List[Tuple[Tuple[float, float], Tuple[float, float], float, float]],
        width: int,
        weight_bias: float,
        side: str,
    ) -> Tuple[List[Tuple[Tuple[float, float], float]], Optional[float]]:
        if not candidates:
            return [], None

        candidates_sorted = sorted(candidates, key=lambda c: c[1][0])
        left_base = candidates_sorted[0][1][0]
        right_base = candidates_sorted[-1][1][0]
        separation = max(
            self.config.boundary_min_separation_px * (width / max(self.config.frame_reference_width, 1e-6)),
            10.0,
        )

        left_points: List[Tuple[Tuple[float, float], float]] = []
        right_points: List[Tuple[Tuple[float, float], float]] = []
        left_slope_sum = 0.0
        left_weight_sum = 0.0
        right_slope_sum = 0.0
        right_weight_sum = 0.0

        for top, bottom, length, vertical_angle in candidates_sorted:
            weight = (length ** self.config.boundary_length_weight_gamma) * (
                1.0 + max(0.0, (bottom[1] - top[1]) / weight_bias)
            )
            slope = (bottom[0] - top[0]) / max(bottom[1] - top[1], 1e-6)

            if bottom[0] - left_base <= separation and vertical_angle >= self.config.boundary_left_min_vertical_angle:
                left_points.append((top, weight))
                left_points.append((bottom, weight))
                left_slope_sum += weight * slope
                left_weight_sum += weight

            if right_base - bottom[0] <= separation and vertical_angle <= self.config.boundary_right_max_vertical_angle:
                right_points.append((top, weight))
                right_points.append((bottom, weight))
                right_slope_sum += weight * slope
                right_weight_sum += weight

        if not left_points and candidates_sorted:
            top, bottom, length, vertical_angle = candidates_sorted[0]
            weight = (length ** self.config.boundary_length_weight_gamma) * (
                1.0 + max(0.0, (bottom[1] - top[1]) / weight_bias)
            )
            left_points.extend([(top, weight), (bottom, weight)])
            slope = (bottom[0] - top[0]) / max(bottom[1] - top[1], 1e-6)
            left_slope_sum += weight * slope
            left_weight_sum += weight

        if not right_points and candidates_sorted:
            top, bottom, length, vertical_angle = candidates_sorted[-1]
            weight = (length ** self.config.boundary_length_weight_gamma) * (
                1.0 + max(0.0, (bottom[1] - top[1]) / weight_bias)
            )
            right_points.extend([(top, weight), (bottom, weight)])
            slope = (bottom[0] - top[0]) / max(bottom[1] - top[1], 1e-6)
            right_slope_sum += weight * slope
            right_weight_sum += weight

        if side == "left":
            slope_hint = (left_slope_sum / left_weight_sum) if left_weight_sum > 0 else None
            return left_points, slope_hint
        else:
            slope_hint = (right_slope_sum / right_weight_sum) if right_weight_sum > 0 else None
            return right_points, slope_hint

    def _fit_boundary(
        self,
        points: List[Tuple[Tuple[float, float], float]],
        slope_hint: Optional[float],
        angle_epsilon: Optional[float] = None,
    ) -> Optional[BoundaryLine]:
        if len(points) < 4:
            return None

        pts = np.array([p for p, _ in points], dtype=np.float64)
        ys = pts[:, 1]
        xs = pts[:, 0]
        weights = np.array([w for _, w in points], dtype=np.float64)
        weights = np.sqrt(np.maximum(weights, 1e-6))

        def _fit(ys_values: np.ndarray, xs_values: np.ndarray, w_values: np.ndarray) -> Tuple[float, float]:
            if w_values.size == 0:
                return tuple(np.polyfit(ys_values, xs_values, 1))
            return tuple(np.polyfit(ys_values, xs_values, 1, w=w_values))

        slope_fit, intercept_fit = _fit(ys, xs, weights)
        residuals = np.abs(slope_fit * ys + intercept_fit - xs)
        median = np.median(residuals)
        if median > 1e-3:
            keep = residuals <= self.config.boundary_outlier_scale * median
            if keep.sum() >= 4:
                slope_fit, intercept_fit = _fit(ys[keep], xs[keep], weights[keep])

        if slope_hint is not None and np.isfinite(slope_hint):
            slope_final = slope_hint
            y_mean = float(np.mean(ys))
            x_mean = float(np.mean(xs))
            intercept_final = x_mean - slope_final * y_mean
        else:
            slope_final = slope_fit
            intercept_final = intercept_fit

        norm = float(np.hypot(slope_final, 1.0))
        if norm < 1e-3:
            return None

        vx = slope_final / norm
        vy = 1.0 / norm
        y0 = float(np.mean(ys))
        x0 = float(slope_final * y0 + intercept_final)

        angle = abs(np.degrees(np.arctan2(vy, vx)))
        if angle_epsilon is None:
            angle_epsilon = self.config.boundary_angle_epsilon
        if angle_epsilon is not None and angle_epsilon >= 0.0:
            upper_limit = 170.0 - angle_epsilon
            if angle < angle_epsilon or angle > upper_limit:
                return None

        return BoundaryLine(float(vx), float(vy), x0, y0)

    def _intersect_lines(
        self,
        left: BoundaryLine,
        right: BoundaryLine,
    ) -> Optional[Tuple[float, float]]:
        a = np.array([[left.vx, -right.vx], [left.vy, -right.vy]], dtype=np.float64)
        b = np.array([right.x0 - left.x0, right.y0 - left.y0], dtype=np.float64)
        det = np.linalg.det(a)
        if abs(det) < 1e-6:
            return None
        t, _ = np.linalg.solve(a, b)
        x = left.x0 + left.vx * t
        y = left.y0 + left.vy * t
        return float(x), float(y)

    def _build_roi_mask(
        self,
        height: int,
        width: int,
        left_line: BoundaryLine,
        right_line: BoundaryLine,
        vanish_row: float,
        expand_px: int,
        tail_px: int,
    ) -> np.ndarray:
        bottom = float(height - 1)
        left_bottom = left_line.x_at(bottom)
        right_bottom = right_line.x_at(bottom)
        if left_bottom is None or right_bottom is None:
            return np.zeros((height, width), dtype=np.uint8)

        top_row = max(0.0, vanish_row - expand_px)
        left_top = left_line.x_at(top_row)
        right_top = right_line.x_at(top_row)
        if left_top is None or right_top is None:
            return np.zeros((height, width), dtype=np.uint8)

        points: List[Tuple[int, int]] = [
            (int(round(left_bottom)), int(round(bottom))),
            (int(round(right_bottom)), int(round(bottom))),
            (int(round(right_top)), int(round(top_row))),
            (int(round(left_top)), int(round(top_row))),
        ]
        tail_row = max(0.0, top_row - self.config.boundary_roi_tail_px)
        tail_row = max(0.0, top_row - tail_px)
        tail_left = left_line.x_at(tail_row)
        tail_right = right_line.x_at(tail_row)
        if tail_left is not None and tail_right is not None and tail_row < top_row:
            points.extend(
                [
                    (int(round(tail_right)), int(round(tail_row))),
                    (int(round(tail_left)), int(round(tail_row))),
                ]
            )
        polygon = np.array(points, dtype=np.int32)
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillConvexPoly(mask, polygon, 255)
        return mask
