from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
import math

from ..types import BoundingBox, LaneSegment
from ..utils import polygon_from_vanish_point
from .base import BaseLaneDetector


@dataclass
class LaneDashDetector(BaseLaneDetector):
    """Detect dashed lane segments using LSD constrained by road boundaries."""

    def detect(
        self,
        frame: np.ndarray,
        vanish_point_row: float,
        vanish_point_col: Optional[float] = None,
        roi_mask: Optional[np.ndarray] = None,
    ) -> List[LaneSegment]:
        height, width = frame.shape[:2]
        vanish_point_row = float(vanish_point_row)
        scale_x = max(width / max(self.config.frame_reference_width, 1e-6), 1e-6)
        scale_y = max(height / max(self.config.frame_reference_height, 1e-6), 1e-6)
        scale_len = (scale_x + scale_y) * 0.5
        min_length = self.config.lane_segment_min_len_px * scale_y
        max_length = self.config.lane_segment_max_len_px * scale_y
        max_single_length = self.config.lane_segment_max_single_len_px * scale_y
        dedup_epsilon = self.config.lane_segment_dedup_epsilon_px * scale_len

        if roi_mask is None:
            roi_mask = self._build_roi_mask(height, width, vanish_point_row)
        if roi_mask is None or not np.any(roi_mask):
            return []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        candidate_mask = self._build_candidate_mask(gray, roi_mask)
        detector = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
        lsd_lines = detector.detect(cv2.bitwise_and(gray, gray, mask=roi_mask))[0]
        hough_lines = cv2.HoughLinesP(
            candidate_mask,
            rho=1,
            theta=np.pi / 180,
            threshold=self.config.lane_hough_threshold,
            minLineLength=self.config.lane_hough_min_length,
            maxLineGap=self.config.lane_hough_max_gap,
        )
        combined_lines: List[np.ndarray] = []
        if lsd_lines is not None:
            combined_lines.append(lsd_lines.reshape(-1, 4))
        if hough_lines is not None:
            combined_lines.append(hough_lines.reshape(-1, 4))
        if not combined_lines:
            return []
        line_array = np.vstack(combined_lines)
        segments: List[LaneSegment] = []
        warm_margin = self.config.lane_roi_margin_px
        min_x_ratio = self.config.lane_column_min_ratio * width
        max_x_ratio = self.config.lane_column_max_ratio * width

        contours, _ = cv2.findContours(candidate_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidate_regions: List[Tuple[int, int, int, int, float]] = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < self.config.lane_min_area_px or area > self.config.lane_max_area_px:
                continue
            x, y, w_box, h_box = cv2.boundingRect(contour)
            length_norm = h_box / float(max(height, 1))
            if not (self.config.lane_segment_length_norm_min <= length_norm <= self.config.lane_segment_length_norm_max):
                continue
            candidate_regions.append((x, y, w_box, h_box, area))

        if not candidate_regions:
            return []

        used: set[Tuple[int, int, int, int]] = set()

        for raw_line in line_array:
            x1, y1, x2, y2 = map(float, raw_line)
            mid_x = (x1 + x2) * 0.5
            mid_y = (y1 + y2) * 0.5
            if mid_y <= vanish_point_row + warm_margin:
                continue

            if not (min_x_ratio <= mid_x <= max_x_ratio):
                continue

            length = float(np.hypot(x2 - x1, y2 - y1))
            if length < min_length or length > max_length:
                continue
            if length > max_single_length:
                continue

            dy = abs(y2 - y1)
            if dy < 1e-3:
                continue

            angle = abs(np.degrees(np.arctan((x2 - x1) / dy)))
            if angle < self.config.lane_vertical_angle_min_deg:
                continue
            if angle > self.config.lane_vertical_angle_deg:
                continue

            brightness = self._sample_line_intensity(gray, (x1, y1), (x2, y2))
            if brightness < self.config.lane_brightness_min_mean:
                continue

            segment_info = self._extract_single_segment(candidate_mask, gray, x1, y1, x2, y2, scale_x)
            if segment_info is None:
                continue
            (sx, sy), (ex, ey), coverage_ratio = segment_info
            refined_length = float(np.hypot(ex - sx, ey - sy))
            if refined_length < min_length or refined_length > max_single_length:
                continue

            min_x = int(np.floor(min(sx, ex)))
            min_y = int(np.floor(min(sy, ey)))
            max_x = int(np.ceil(max(sx, ex)))
            max_y = int(np.ceil(max(sy, ey)))
            w_box = max(1, max_x - min_x)
            h_box = max(1, max_y - min_y)

            length_norm = h_box / float(max(height, 1))
            if not (self.config.lane_segment_length_norm_min <= length_norm <= self.config.lane_segment_length_norm_max):
                continue

            min_row_allowed = vanish_point_row + height * self.config.lane_vanish_min_relative_row
            center_y = (sy + ey) * 0.5
            if center_y < min_row_allowed:
                continue

            best_density = 0.0
            for x, y, w_region, h_region, _ in candidate_regions:
                x_end = min(x + w_region, max_x)
                y_end = min(y + h_region, max_y)
                x_start = max(x, min_x)
                y_start = max(y, min_y)
                if x_end <= x_start or y_end <= y_start:
                    continue
                slice_mask = candidate_mask[y_start:y_end, x_start:x_end]
                if slice_mask.size == 0:
                    continue
                density = float(cv2.countNonZero(slice_mask)) / float(slice_mask.size)
                if density > best_density:
                    best_density = density

            if best_density < self.config.lane_segment_min_density:
                continue

            if vanish_point_col is not None:
                center_x = (sx + ex) * 0.5
                center_y = (sy + ey) * 0.5
                direction = np.array([ex - sx, ey - sy], dtype=np.float64)
                vanish_vec = np.array([vanish_point_col - center_x, vanish_point_row - center_y], dtype=np.float64)
                norm_dir = float(np.linalg.norm(direction))
                norm_vanish = float(np.linalg.norm(vanish_vec))
                if norm_dir < 1e-3 or norm_vanish < 1e-3:
                    continue
                cosine = abs(float(np.dot(direction, vanish_vec)) / (norm_dir * norm_vanish))
                cosine = min(1.0, max(-1.0, cosine))
                angle_to_vanish = float(np.degrees(np.arccos(cosine)))
                if angle_to_vanish > self.config.lane_vanish_alignment_deg:
                    continue

            key = (min_x, min_y, max_x, max_y)
            if key in used:
                continue
            used.add(key)

            contour_line = np.array([[sx, sy], [ex, ey]], dtype=np.float32)
            box = BoundingBox(
                x=min_x,
                y=min_y,
                width=w_box,
                height=h_box,
                confidence=coverage_ratio,
                label="lane_dash",
            )
            segments.append(LaneSegment(contour=contour_line.astype(np.int32), bounding_box=box, length_px=refined_length))

        segments = self._deduplicate_segments(segments, dedup_epsilon)
        segments.sort(key=lambda seg: seg.bounding_box.bottom())
        return segments

    def _build_candidate_mask(self, gray: np.ndarray, roi_mask: np.ndarray) -> np.ndarray:
        clahe = cv2.createCLAHE(
            clipLimit=max(0.0, self.config.lane_clahe_clip),
            tileGridSize=(max(1, self.config.lane_clahe_grid), max(1, self.config.lane_clahe_grid)),
        )
        equalized = clahe.apply(gray)
        kernel_size = max(3, self.config.lane_tophat_kernel | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        tophat = cv2.morphologyEx(equalized, cv2.MORPH_TOPHAT, kernel)
        blended = cv2.addWeighted(
            tophat,
            self.config.lane_tophat_weight,
            equalized,
            1.0 - self.config.lane_tophat_weight,
            0.0,
        )

        block_size = max(3, self.config.lane_adaptive_threshold_block | 1)
        threshold_c = self.config.lane_adaptive_threshold_c
        adaptive = cv2.adaptiveThreshold(
            blended,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            threshold_c,
        )
        _, otsu = cv2.threshold(blended, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, bright = cv2.threshold(equalized, self.config.lane_min_brightness, 255, cv2.THRESH_BINARY)
        combined = cv2.bitwise_or(adaptive, otsu)
        combined = cv2.bitwise_or(combined, bright)
        combined = cv2.bitwise_and(combined, roi_mask)

        vertical_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (3, max(3, self.config.lane_vertical_kernel | 1)),
        )
        processed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, vertical_kernel, iterations=1)
        processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, vertical_kernel, iterations=1)
        processed = cv2.dilate(processed, vertical_kernel, iterations=1)
        return processed

    def _build_roi_mask(self, height: int, width: int, vanish_point_row: float) -> np.ndarray:
        polygon = polygon_from_vanish_point(
            width=width,
            height=height,
            vanish_point_row=vanish_point_row,
            top_ratio=self.config.roi_top_ratio,
            bottom_ratio=self.config.roi_bottom_ratio,
            expand_px=self.config.roi_expand,
        )
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillConvexPoly(mask, np.array(polygon, dtype=np.int32), 255)
        return mask

    def _extract_single_segment(
        self,
        mask: np.ndarray,
        gray: np.ndarray,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        scale_x: float,
    ) -> Optional[Tuple[Tuple[float, float], Tuple[float, float], float]]:
        length = float(math.hypot(x2 - x1, y2 - y1))
        samples = max(int(round(length)), 40)
        if samples < 2:
            return None
        xs = np.linspace(x1, x2, samples)
        ys = np.linspace(y1, y2, samples)
        hits = []
        height, width = mask.shape[:2]
        direction = np.array([x2 - x1, y2 - y1], dtype=np.float64)
        norm_dir = np.linalg.norm(direction)
        if norm_dir < 1e-6:
            return None
        direction /= norm_dir
        perp = np.array([-direction[1], direction[0]], dtype=np.float64)
        base_offset = self.config.lane_sample_width * max(scale_x, 1.0)
        offsets = [base_offset * 1.5, base_offset * 3.0]

        for xi, yi in zip(xs, ys):
            ix = int(round(xi))
            iy = int(round(yi))
            if ix < 0 or iy < 0 or ix >= width or iy >= height:
                hits.append(False)
                continue
            mask_hit = mask[iy, ix] > 0
            center = float(gray[iy, ix])
            background_values: List[float] = []
            for off in offsets:
                px = int(round(ix + perp[0] * off))
                py = int(round(iy + perp[1] * off))
                if 0 <= px < width and 0 <= py < height:
                    background_values.append(float(gray[py, px]))
            background = float(np.mean(background_values)) if background_values else center
            contrast = center - background
            hits.append(mask_hit or contrast >= self.config.lane_contrast_threshold)

        total = len(hits)
        if total == 0:
            return None

        min_cont = max(1, int(total * self.config.lane_segment_min_continuous_ratio))
        max_gap = max(1, int(total * self.config.lane_segment_max_gap_ratio))

        segments_idx: List[Tuple[int, int]] = []
        start = None
        gap = 0
        for idx, hit in enumerate(hits):
            if hit:
                if start is None:
                    start = idx
                gap = 0
            else:
                if start is not None:
                    gap += 1
                    if gap > max_gap:
                        end = idx - gap
                        if end >= start and (end - start + 1) >= min_cont:
                            segments_idx.append((start, end))
                        start = None
                        gap = 0
        if start is not None:
            end = total - 1 - gap if gap > 0 else total - 1
            if end >= start and (end - start + 1) >= min_cont:
                segments_idx.append((start, end))

        if not segments_idx:
            return None

        segments_idx.sort(key=lambda pair: pair[1] - pair[0], reverse=True)
        primary = segments_idx[0]
        if len(segments_idx) > 1 and (segments_idx[1][1] - segments_idx[1][0]) >= min_cont:
            return None

        start_ratio = primary[0] / (samples - 1)
        end_ratio = primary[1] / (samples - 1)
        sx = x1 + (x2 - x1) * start_ratio
        sy = y1 + (y2 - y1) * start_ratio
        ex = x1 + (x2 - x1) * end_ratio
        ey = y1 + (y2 - y1) * end_ratio
        coverage = float(primary[1] - primary[0] + 1) / float(total)
        return (sx, sy), (ex, ey), coverage

    def _deduplicate_segments(self, segments: List[LaneSegment], epsilon: float) -> List[LaneSegment]:
        if not segments:
            return []
        epsilon_sq = epsilon ** 2
        kept: List[LaneSegment] = []
        centers: List[Tuple[float, float]] = []
        for seg in sorted(segments, key=lambda s: -s.length_px):
            start = seg.contour[0]
            end = seg.contour[-1]
            center = ((start[0] + end[0]) * 0.5, (start[1] + end[1]) * 0.5)
            if all((center[0] - cx) ** 2 + (center[1] - cy) ** 2 >= epsilon_sq for cx, cy in centers):
                kept.append(seg)
                centers.append(center)
        return kept

    def _sample_line_intensity(
        self,
        image: np.ndarray,
        start: Tuple[float, float],
        end: Tuple[float, float],
    ) -> float:
        x1, y1 = start
        x2, y2 = end
        thickness = max(1, self.config.lane_sample_width)
        probe = np.zeros_like(image, dtype=np.uint8)
        cv2.line(probe, (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2))), 255, thickness=thickness)
        values = image[probe == 255]
        if values.size == 0:
            return 0.0
        return float(np.mean(values))
