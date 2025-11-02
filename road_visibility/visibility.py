from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .background import BackgroundEdgeModel
from .camera_model import distance_from_row, lambda_from_segment
from .config import PipelineConfig
from .detectors import BaseVehicleDetector, LaneBoundaryDetector, LaneDashDetector
from .types import BoundingBox, LaneSegment, VisibilityEstimate
from .utils import clamp, count_edges, dilate_erode, ensure_directory, polygon_from_vanish_point, trim_mean


@dataclass
class RoadVisibilityEstimator:
    config: PipelineConfig
    vehicle_detectors: Optional[List[BaseVehicleDetector]] = None

    def __post_init__(self) -> None:
        if self.vehicle_detectors is None:
            self.vehicle_detectors = []
        elif isinstance(self.vehicle_detectors, BaseVehicleDetector):
            self.vehicle_detectors = [self.vehicle_detectors]

        self.boundary_detector = LaneBoundaryDetector(self.config)
        self.lane_detector = LaneDashDetector(self.config)
        self.background = BackgroundEdgeModel(self.config)
        self.vehicle_upboard_history: deque[float] = deque(maxlen=self.config.vehicle_history_size)
        self.queue_compare: deque[float] = deque(maxlen=self.config.queue_size)
        self.frame_counter: int = 0
        self.locked_vanish_point: Optional[Tuple[float, float]] = None
        self.vanish_outlier_count: int = 0
        self.reference_clear_frame: Optional[np.ndarray] = None
        self.reference_dark_channel: Optional[np.ndarray] = None
        self.reference_transmittance_mean: Optional[float] = None
        self.locked_lambda: Optional[float] = None
        self.lambda_outlier_count: int = 0
        self.last_lambda_candidate: Optional[float] = None
        self.vehicle_column_history: Optional[np.ndarray] = None
        self.last_vehicle_visibility: Optional[float] = None
        self.reference_lambda: Optional[float] = None
        self.reference_lambda_base: Optional[float] = None
        self._cached_vehicle_boxes: List[BoundingBox] = []
        self.last_trans_row: Optional[float] = None
        self.last_compare_row: Optional[float] = None
        self.use_transmittance_video_fusion: bool = False
        self._trans_fusion_buffer: deque[np.ndarray] = deque(
            maxlen=max(self.config.transmittance_video_window, 1)
        )
        self._trans_fusion_frame: Optional[np.ndarray] = None
        self._suppress_fusion_update: bool = False
        self.smoothed_compare: Optional[float] = None
        self.smoothed_trans: Optional[float] = None
        self.fused_visibility: Optional[float] = None
        self.vehicle_upper_bound: Optional[float] = None
        self.vehicle_upper_bound_missing: int = 0
        self.vehicle_detect_history: deque[float] = deque(
            maxlen=max(int(self.config.vehicle_upper_bound_window), 1)
        )

    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        target_w = max(int(self.config.frame_target_width), 0)
        target_h = max(int(self.config.frame_target_height), 0)
        if target_w > 0 and target_h > 0 and (frame.shape[1] != target_w or frame.shape[0] != target_h):
            return cv2.resize(frame, (target_w, target_h))
        return frame

    def initialize(
        self,
        clear_frame: np.ndarray,
        vehicle_boxes: Optional[Sequence[BoundingBox]] = None,
    ) -> VisibilityEstimate:
        clear_frame = self._resize_frame(clear_frame)
        self.frame_counter = 0
        self.vehicle_upboard_history.clear()
        self.queue_compare.clear()
        self.background = BackgroundEdgeModel(self.config)
        self.vehicle_column_history = None
        self._cached_vehicle_boxes = []
        self._trans_fusion_buffer.clear()
        self._trans_fusion_frame = None
        approx_row = clear_frame.shape[0] * self.config.roi_top_ratio
        boundary_detection = self.boundary_detector.detect(clear_frame, approx_row)
        if boundary_detection.vanish_point is not None:
            vp_x, vp_y = boundary_detection.vanish_point
        else:
            vp_x = clear_frame.shape[1] * 0.5
            vp_y = approx_row
        self.locked_vanish_point = (vp_x, vp_y)
        self.vanish_outlier_count = 0
        self.locked_lambda = None
        self.lambda_outlier_count = 0
        self.last_lambda_candidate = None
        self.reference_clear_frame = clear_frame.copy()
        clear_float = clear_frame.astype(np.float32)
        dark_clear = self._dark_channel(clear_float)
        self.reference_dark_channel = np.maximum(dark_clear, 1.0)
        self.reference_transmittance_mean = None
        self.last_vehicle_visibility = None
        self.reference_lambda = None
        self.reference_lambda_base = None
        self.last_trans_row = None
        self.smoothed_compare = None
        self.smoothed_trans = None
        self.fused_visibility = None
        self.vehicle_upper_bound = None
        self.vehicle_upper_bound_missing = 0
        self.vehicle_detect_history.clear()

        self._suppress_fusion_update = True
        visibility = self.estimate(clear_frame, vehicle_boxes=vehicle_boxes)
        if visibility.mean_transmittance is not None:
            self.reference_transmittance_mean = visibility.mean_transmittance
        else:
            self.reference_transmittance_mean = None
        self._suppress_fusion_update = False
        return visibility

    def warmup(
        self,
        clear_frame: np.ndarray,
        additional_frames: Optional[Iterable[np.ndarray]] = None,
        vehicle_boxes: Optional[Sequence[BoundingBox]] = None,
    ) -> VisibilityEstimate:
        visibility = self.initialize(clear_frame, vehicle_boxes=vehicle_boxes)
        if additional_frames is None:
            return visibility
        for frame in additional_frames:
            self.estimate(frame, vehicle_boxes=None)
        return visibility

    def estimate(
        self,
        frame: np.ndarray,
        vehicle_boxes: Optional[Sequence[BoundingBox]] = None,
    ) -> VisibilityEstimate:
        frame = self._resize_frame(frame)
        if self.use_transmittance_video_fusion and not self._suppress_fusion_update:
            window = max(self.config.transmittance_video_window, 1)
            if window > 0:
                self._trans_fusion_buffer.append(frame.copy())
                if len(self._trans_fusion_buffer) == window:
                    current_index = self.frame_counter + 1
                    if current_index % window == 0 or self._trans_fusion_frame is None:
                        stack = np.stack(self._trans_fusion_buffer, axis=0).astype(np.float32)
                        avg = np.mean(stack, axis=0)
                        self._trans_fusion_frame = np.clip(avg, 0.0, 255.0).astype(np.uint8)
        self.frame_counter += 1
        if self.locked_vanish_point is not None:
            approx_vanish_row = clamp(
                self.locked_vanish_point[1],
                -self.config.vanish_point_offset_limit,
                frame.shape[0] - 1,
            )
        else:
            approx_vanish_row = frame.shape[0] * self.config.roi_top_ratio
        boundary_detection = self.boundary_detector.detect(frame, approx_vanish_row)
        roi_mask = boundary_detection.roi_mask
        use_segment_refine = True
        if boundary_detection.vanish_point is not None:
            self._update_vanish_point_lock(*boundary_detection.vanish_point)
            use_segment_refine = False

        if self.locked_vanish_point is None:
            self.locked_vanish_point = (frame.shape[1] * 0.5, approx_vanish_row)
        vp_x, vp_y = self.locked_vanish_point
        vanish_point_row = clamp(
            vp_y,
            -self.config.vanish_point_offset_limit,
            frame.shape[0] - 1,
        )
        vanish_point_col = clamp(vp_x, 0, frame.shape[1] - 1)
        roi_mask = self._expand_roi_with_vehicle_history(roi_mask, vanish_point_row, frame.shape[:2])
        edges = cv2.Canny(frame, *self.config.canny_thresholds)
        detection_interval = max(1, self.config.vehicle_detection_interval)
        if vehicle_boxes is None:
            should_detect = (
                bool(self.vehicle_detectors)
                and (
                    not self._cached_vehicle_boxes
                    or (self.frame_counter - 1) % detection_interval == 0
                )
            )
            if should_detect:
                detected = self._run_vehicle_detectors(frame)
                self._cached_vehicle_boxes = list(detected)
            vehicle_boxes = list(self._cached_vehicle_boxes)
        else:
            vehicle_boxes = list(vehicle_boxes)
            self._cached_vehicle_boxes = list(vehicle_boxes)
        self._update_vehicle_history(vehicle_boxes, frame.shape[1])

        road_mask: Optional[np.ndarray]
        if roi_mask is not None:
            road_mask = roi_mask
        else:
            polygon = polygon_from_vanish_point(
                frame.shape[1],
                frame.shape[0],
                vanish_point_row,
                self.config.roi_top_ratio,
                self.config.roi_bottom_ratio,
                self.config.roi_expand,
            )
            road_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.fillConvexPoly(road_mask, np.array(polygon, dtype=np.int32), 255)

        masked_edges = edges.copy()
        if vehicle_boxes:
            h, w = edges.shape[:2]
            for box in vehicle_boxes:
                x1 = int(clamp(box.x, 0, w - 1))
                y1 = int(clamp(box.y, 0, h - 1))
                x2 = int(clamp(box.x + box.width, 0, w))
                y2 = int(clamp(box.y + box.height, 0, h))
                masked_edges[y1:y2, x1:x2] = 0
        if road_mask is not None:
            masked_edges = cv2.bitwise_and(masked_edges, road_mask)
        self.background.update(masked_edges)

        lane_segments = self.lane_detector.detect(
            frame,
            vanish_point_row,
            vanish_point_col=vanish_point_col,
            roi_mask=roi_mask,
        )
        if use_segment_refine:
            self._refine_vanish_with_segments(lane_segments, vanish_point_row, frame.shape[1])
            vp_x, vp_y = self.locked_vanish_point
            vanish_point_row = clamp(
                vp_y,
                -self.config.vanish_point_offset_limit,
                frame.shape[0] - 1,
            )
            vanish_point_col = clamp(vp_x, 0, frame.shape[1] - 1)

        lambda_lane = self._lambda_from_lane_segments(lane_segments, vanish_point_row)
        lambda_vehicle = self._lambda_from_vehicles(vehicle_boxes, vanish_point_row)

        lambda_fused = self._fuse_lambda(lambda_lane, lambda_vehicle)

        visibility_detect = self._visibility_from_vehicles(lambda_fused, vehicle_boxes, vanish_point_row)
        visibility_compare, _ = self._visibility_from_edges(
            frame,
            edges,
            lambda_fused,
            vanish_point_row,
            roi_mask,
        )
        # cv2.imwrite("1_mask.png",roi_mask)
        mean_transmittance: Optional[float] = None
        trans_hist: Optional[np.ndarray] = None
        trans_bins: Optional[np.ndarray] = None
        visibility_trans: Optional[float] = None
        trans_percentile: Optional[float] = None
        if self.reference_dark_channel is not None:
            (
                trans_hist,
                trans_bins,
                mean_transmittance,
                visibility_trans,
                trans_percentile,
            ) = self.compute_transmittance_metrics(
                frame,
                roi_mask,
                vanish_point_row,
                lambda_fused,
            )

        visibility_fused = self._compute_fused_visibility(
            visibility_compare,
            visibility_trans,
            visibility_detect,
        )

        estimate = VisibilityEstimate(
            vanish_point_row=vanish_point_row,
            vanish_point_col=vanish_point_col,
            lambda_from_lane=lambda_lane,
            lambda_from_vehicle=lambda_vehicle,
            lambda_fused=lambda_fused,
            visibility_detect=visibility_detect,
            visibility_compare=visibility_compare,
            roi_mask=roi_mask,
            lane_segments=lane_segments,
            vehicle_boxes=vehicle_boxes,
            mean_transmittance=mean_transmittance,
            transmittance_histogram=trans_hist,
            transmittance_bins=trans_bins,
            visibility_transmittance=visibility_trans,
            transmittance_percentile_value=trans_percentile,
            visibility_fused=visibility_fused,
        )

        return estimate

    def compute_transmittance_metrics(
        self,
        frame: np.ndarray,
        roi_mask: Optional[np.ndarray],
        vanish_point_row: float,
        lambda_value: float,
        bins: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, float, Optional[float], Optional[float]]:
        if self.reference_dark_channel is None:
            raise ValueError("Estimator not initialised with a clear reference frame.")

        frame_for_trans = frame
        if self.use_transmittance_video_fusion and self._trans_fusion_frame is not None:
            frame_for_trans = self._trans_fusion_frame
        trans_map = self._compute_transmittance_map(frame_for_trans)
        if (
            self.reference_transmittance_mean is not None
            and self.reference_transmittance_mean > 1e-6
        ):
            scale = float(self.reference_transmittance_mean)
            trans_map = np.clip(trans_map / scale, 0.0, 1.0).astype(np.float32)

        if roi_mask is not None and roi_mask.shape == trans_map.shape:
            if roi_mask.dtype != np.uint8:
                mask_uint8 = np.where(roi_mask > 0, 1, 0).astype(np.uint8)
            else:
                mask_uint8 = roi_mask
            mask_bool = mask_uint8.astype(bool)
        else:
            fallback_mask = np.zeros_like(trans_map, dtype=np.uint8)
            polygon = polygon_from_vanish_point(
                frame.shape[1],
                frame.shape[0],
                vanish_point_row,
                self.config.roi_top_ratio,
                self.config.roi_bottom_ratio,
                self.config.roi_expand,
            )
            cv2.fillConvexPoly(fallback_mask, np.array(polygon, dtype=np.int32), 1)
            mask_uint8 = fallback_mask
            mask_bool = fallback_mask.astype(bool)

        values = trans_map[mask_bool]
        if values.size < self.config.transmittance_visibility_min_pixels:
            return (
                np.zeros(1, dtype=np.float32),
                np.linspace(0.0, 1.0, num=2),
                0.0,
                None,
                None,
            )

        hist_bins = bins or self.config.transmittance_hist_bins
        hist, bin_edges = np.histogram(values, bins=hist_bins, range=(0.0, 1.0))
        hist = hist.astype(np.float32)
        if hist.sum() > 0:
            hist /= hist.sum()
        smooth = int(self.config.transmittance_hist_smoothing)
        if smooth > 1 and hist.size:
            kernel = np.ones(smooth, dtype=np.float32)
            kernel /= kernel.sum()
            hist = np.convolve(hist, kernel, mode="same")
            if hist.sum() > 0:
                hist /= hist.sum()
        mean = float(np.mean(values)) if values.size else 0.0
        visibility, percentile_value = self._compute_transmittance_visibility(
            trans_map,
            mask_uint8,
            vanish_point_row,
            lambda_value,
        )
        return hist, bin_edges, mean, visibility, percentile_value

    def is_clear_scene(self, mean_transmittance: Optional[float]) -> bool:
        if mean_transmittance is None:
            return False
        return mean_transmittance >= self.config.clear_scene_transmittance_threshold

    def _compute_transmittance_map(self, frame: np.ndarray) -> np.ndarray:
        if self.reference_dark_channel is None:
            raise ValueError("Estimator not initialised with a clear reference frame.")
        frame_float = frame.astype(np.float32)
        dark_fog = self._dark_channel(frame_float)

        reference_dc = self.reference_dark_channel
        ratio_reference = np.minimum(dark_fog, reference_dc) / (reference_dc + 1e-6)
        ratio_reference = np.clip(ratio_reference, 0.0, 1.0)

        atmosphere = self._estimate_atmospheric_light(frame_float, dark_fog)
        atmosphere = np.maximum(atmosphere.astype(np.float32), 1.0)
        norm = frame_float / (atmosphere.reshape(1, 1, 3) + 1e-6)
        ratio_atmosphere = np.min(norm, axis=2)
        ratio_atmosphere = np.clip(ratio_atmosphere, 0.0, 1.0)

        ratio = np.minimum(ratio_reference, ratio_atmosphere)

        trans_map = 1.0 - self.config.transmittance_dcp_omega * ratio
        trans_map = np.clip(trans_map, 0.0, 1.0).astype(np.float32)

        sigma = max(self.config.transmittance_gaussian_sigma, 0.0)
        if sigma > 1e-6:
            trans_map = cv2.GaussianBlur(trans_map, (0, 0), sigmaX=sigma, sigmaY=sigma)

        low_frac = clamp(self.config.transmittance_contrast_low, 0.0, 1.0)
        high_frac = clamp(self.config.transmittance_contrast_high, 0.0, 1.0)
        if high_frac <= low_frac:
            high_frac = min(1.0, low_frac + 1e-3)
        if trans_map.size > 0 and high_frac > low_frac:
            low_val = float(np.percentile(trans_map, low_frac * 100.0))
            high_val = float(np.percentile(trans_map, high_frac * 100.0))
            if high_val > low_val + 1e-6:
                normalized = (trans_map - low_val) / (high_val - low_val)
                normalized = np.clip(normalized, 0.0, 1.0)
                blend = clamp(self.config.transmittance_contrast_blend, 0.0, 1.0)
                trans_map = (1.0 - blend) * trans_map + blend * normalized

        gamma = max(self.config.transmittance_gamma, 1e-3)
        if abs(gamma - 1.0) > 1e-6:
            trans_map = np.power(trans_map, gamma)
        return trans_map

    def _estimate_atmospheric_light(self, frame: np.ndarray, dark_channel: np.ndarray) -> np.ndarray:
        percent = clamp(self.config.transmittance_atmosphere_top_percent, 0.0, 1.0)
        if percent <= 0.0:
            percent = 1e-3
        flat_dark = dark_channel.reshape(-1)
        pixel_count = flat_dark.size
        sample_count = max(1, int(round(pixel_count * percent)))
        indices = np.argpartition(flat_dark, -sample_count)[-sample_count:]
        frame_flat = frame.reshape(-1, frame.shape[2])
        atmosphere = frame_flat[indices].max(axis=0)
        return atmosphere

    def _compute_transmittance_visibility(
        self,
        trans_map: np.ndarray,
        roi_mask: np.ndarray,
        vanish_point_row: float,
        lambda_value: float,
    ) -> Tuple[Optional[float], Optional[float]]:
        self.last_trans_row = None
        if trans_map.size == 0:
            return None, None

        combined_mask = roi_mask.astype(bool)
        if not np.any(combined_mask):
            return None, None

        values = trans_map[combined_mask]
        if values.size < self.config.transmittance_visibility_min_pixels:
            return None, None

        percentile = clamp(self.config.transmittance_visibility_percentile, 0.0, 1.0) * 100.0
        pixel_percentile_value = float(np.percentile(values, percentile))

        row_counts = np.sum(combined_mask, axis=1)
        row_sums = np.sum(trans_map * combined_mask, axis=1)
        row_means = np.zeros_like(row_counts, dtype=np.float32)
        valid_rows = row_counts > 0
        row_means[valid_rows] = row_sums[valid_rows] / (row_counts[valid_rows] + 1e-6)

        valid_indices = np.where(valid_rows)[0]
        if valid_indices.size == 0:
            return None, None
        base_fraction = clamp(self.config.transmittance_row_base_fraction, 0.0, 1.0)
        base_fraction = max(base_fraction, 1e-3)
        base_count = int(np.ceil(valid_indices.size * base_fraction))
        base_count = max(base_count, 1)
        base_slice = valid_indices[:base_count]
        base_mean = float(np.mean(row_means[base_slice]))

        threshold_weight = clamp(self.config.transmittance_row_threshold_scale, 0.0, 1.0)
        row_threshold_means = float(np.percentile(row_means[valid_rows], percentile))
        row_max = float(np.max(row_means[valid_rows]))

        target_threshold = pixel_percentile_value + (row_threshold_means - pixel_percentile_value) * threshold_weight
        lower_bound = min(pixel_percentile_value, row_max)
        upper_bound = max(pixel_percentile_value, row_max)
        if row_max >= lower_bound:
            target_threshold = float(np.clip(target_threshold, lower_bound, row_max))
        else:
            target_threshold = float(np.clip(target_threshold, row_max, upper_bound))

        if target_threshold <= base_mean + 1e-6:
            row_threshold = float(np.clip(target_threshold, 0.0, 1.0))
        else:
            progress = clamp(self.config.transmittance_row_threshold_blend, 0.0, 1.0)
            row_threshold = base_mean + (target_threshold - base_mean) * progress
            row_threshold = max(row_threshold, base_mean)
            row_threshold = min(row_threshold, target_threshold)
            row_threshold = float(np.clip(row_threshold, 0.0, 1.0))

        row_mask = row_means >= row_threshold
        candidate_rows = np.where(row_mask)[0]
        if candidate_rows.size == 0:
            return None, row_threshold

        selected_row = float(np.min(candidate_rows))
        self.last_trans_row = selected_row
        distance = distance_from_row(lambda_value, vanish_point_row, selected_row)
        # distance *= max(row_threshold, 0.0)
        return float(distance), row_threshold

    def _dark_channel(self, image: np.ndarray) -> np.ndarray:
        radius = max(int(self.config.transmittance_dcp_radius), 1)
        kernel_size = radius * 2 + 1
        min_rgb = np.min(image, axis=2)
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        return cv2.erode(min_rgb, kernel)

    def _lambda_from_lane_segments(
        self,
        segments: Sequence[LaneSegment],
        vanish_point_row: float,
    ) -> Optional[float]:
        lambdas: List[float] = []
        for seg in segments:
            top = seg.bounding_box.top()
            bottom = seg.bounding_box.bottom()
            value = lambda_from_segment(
                top,
                bottom,
                vanish_point_row,
                self.config.lane_dash_length_m,
            )
            if value is not None:
                lambdas.append(value)

        if not lambdas:
            return None

        fused = trim_mean(lambdas, self.config.lambda_trim_fraction)
        if np.isfinite(fused):
            return float(fused)
        return None

    def _run_vehicle_detectors(self, frame: np.ndarray) -> List[BoundingBox]:
        results: List[BoundingBox] = []
        for detector in self.vehicle_detectors or []:
            results.extend(detector.detect(frame))
        return results

    def _lambda_from_vehicles(
        self,
        vehicle_boxes: Sequence[BoundingBox],
        vanish_point_row: float,
    ) -> Optional[float]:
        lambdas: List[float] = []
        for box in vehicle_boxes:
            if box.bottom() <= vanish_point_row:
                continue
            value = lambda_from_segment(
                box.top(),
                box.bottom(),
                vanish_point_row,
                self.config.vehicle_length_m,
            )
            if value is not None:
                lambdas.append(value)

        if not lambdas:
            return None

        fused = trim_mean(lambdas, self.config.lambda_trim_fraction)
        if np.isfinite(fused):
            return fused
        return None

    def _fuse_lambda(self, lambda_lane: Optional[float], lambda_vehicle: Optional[float]) -> float:
        if self.reference_lambda is None:
            base = None
            if lambda_lane is not None and np.isfinite(lambda_lane):
                base = float(lambda_lane)
            elif lambda_vehicle is not None and np.isfinite(lambda_vehicle):
                base = float(lambda_vehicle)
            if base is None:
                base = self.config.default_lambda
            self.reference_lambda = base
            self.reference_lambda_base = base
            self.locked_lambda = base
            self.last_lambda_candidate = base
            return base

        base = self.reference_lambda_base if self.reference_lambda_base is not None else self.reference_lambda
        current = self.locked_lambda if self.locked_lambda is not None else self.reference_lambda
        current = float(current)

        candidate: Optional[float] = None
        if lambda_vehicle is not None and np.isfinite(lambda_vehicle):
            candidate = float(lambda_vehicle)
            max_dev = max(self.config.lambda_vehicle_max_deviation, 0.0)
            lower = base * (1.0 - max_dev)
            upper = base * (1.0 + max_dev)
            candidate = float(np.clip(candidate, lower, upper))
            alpha = clamp(self.config.lambda_vehicle_update_alpha, 0.0, 1.0)
            current = (1.0 - alpha) * current + alpha * candidate
            current = float(np.clip(current, lower, upper))

        self.reference_lambda = current
        self.locked_lambda = current
        self.last_lambda_candidate = candidate
        self.lambda_outlier_count = 0
        return current

    def _visibility_from_vehicles(
        self,
        lambda_value: float,
        vehicle_boxes: Sequence[BoundingBox],
        vanish_point_row: float,
    ) -> Optional[float]:
        if not vehicle_boxes:
            return None

        sorted_boxes = sorted(vehicle_boxes, key=lambda b: b.top())
        top_row = sorted_boxes[0].top()
        top_row = max(top_row, vanish_point_row + 10)
        self.vehicle_upboard_history.append(top_row)
        filtered_top = min(self.vehicle_upboard_history) if self.vehicle_upboard_history else top_row

        distance = lambda_value / max(filtered_top - vanish_point_row + 0.1, 0.1)
        self.last_vehicle_visibility = distance
        return float(distance)

    def _visibility_from_edges(
        self,
        frame: np.ndarray,
        edges: np.ndarray,
        lambda_value: float,
        vanish_point_row: float,
        roi_mask: Optional[np.ndarray],
    ) -> Tuple[float, Optional[np.ndarray]]:
        reference = self.background.get_reference()
        self.last_compare_row = None
        if roi_mask is not None and roi_mask.shape == edges.shape:
            if roi_mask.dtype == np.uint8:
                base_mask = roi_mask.copy()
            else:
                base_mask = np.where(roi_mask > 0, 255, 0).astype(np.uint8)
        else:
            base_mask = np.zeros_like(edges, dtype=np.uint8)
            polygon = polygon_from_vanish_point(
                frame.shape[1],
                frame.shape[0],
                vanish_point_row,
                self.config.roi_top_ratio,
                self.config.roi_bottom_ratio,
                self.config.roi_expand,
            )
            cv2.fillConvexPoly(base_mask, np.array(polygon, dtype=np.int32), 255)
        # if roi_mask is not None:
            # cv2.imwrite("1_mask.png",roi_mask)
        if reference is None:
            fallback_delta = max(10.0, frame.shape[0] * 0.5 - vanish_point_row)
            distance = lambda_value / (0.1 + fallback_delta)
            self.queue_compare.append(distance)
            return float(min(self.queue_compare)), base_mask

        expand_ratio = max(self.config.vis_compare_roi_expand_ratio, 0.0)
        working_mask = base_mask
        if expand_ratio > 1e-6:
            working_mask = base_mask.copy()
            kernel_size = int(max(frame.shape[0], frame.shape[1]) * expand_ratio)
            kernel_size = max(3, kernel_size)
            if kernel_size % 2 == 0:
                kernel_size += 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            working_mask = cv2.dilate(working_mask, kernel, iterations=1)

        edges_masked = cv2.bitwise_and(edges, working_mask)
        reference_masked = cv2.bitwise_and(reference, working_mask)

        edges_processed = dilate_erode(edges_masked, self.config.morph_kernel_size)
        reference_processed = dilate_erode(reference_masked, self.config.morph_kernel_size)

        row_start = int(max(0, round(vanish_point_row)))
        first_visible = -1
        stable = 0

        for y in range(row_start, edges.shape[0] - self.config.edge_window_rows):
            row_range = slice(y, y + self.config.edge_window_rows)
            ref_slice = reference_processed[row_range, :]
            curr_slice = edges_processed[row_range, :]

            ref_edges = count_edges(ref_slice)
            if ref_edges < self.config.edge_min_reference_edges:
                stable = 0
                continue

            curr_edges = count_edges(curr_slice)
            retention = curr_edges / (ref_edges + 1e-3)
            if retention >= self.config.edge_retention_threshold:
                stable += 1
                if stable >= self.config.edge_required_stable:
                    first_visible = y - self.config.edge_required_stable // 2
                    break
            else:
                stable = 0

        if first_visible < 0:
            first_visible = edges.shape[0] - 1
        min_row = int(
            max(
                vanish_point_row + self.config.edge_visibility_min_delta_rows,
                row_start,
            )
        )
        min_row = min(min_row, edges.shape[0] - 1)
        if first_visible < min_row:
            first_visible = min_row

        self.last_compare_row = float(first_visible)
        distance = distance_from_row(lambda_value, vanish_point_row, first_visible)
        distance = min(distance, self.config.visibility_compare_max_distance)
        self.queue_compare.append(distance)
        return float(min(self.queue_compare)), base_mask

    def _update_vehicle_upper_bound(self, detection: Optional[float]) -> Optional[float]:
        if detection is not None and detection > 0.0:
            detection_clamped = max(detection, self.config.visibility_min_distance)
            self.vehicle_detect_history.append(float(detection_clamped))
            if self.vehicle_detect_history:
                self.vehicle_upper_bound = float(min(self.vehicle_detect_history))
            self.vehicle_upper_bound_missing = 0
        else:
            self.vehicle_detect_history.clear()
            self.vehicle_upper_bound_missing += 1
            if self.vehicle_upper_bound is not None:
                relax = max(self.config.vehicle_upper_bound_relax, 1.0)
                self.vehicle_upper_bound *= relax
                max_missing = max(int(self.config.vehicle_upper_bound_window), 1)
                if self.vehicle_upper_bound_missing >= max_missing:
                    self.vehicle_upper_bound = None

        if self.vehicle_upper_bound is not None:
            self.vehicle_upper_bound = float(
                max(
                    self.config.visibility_min_distance,
                    min(self.vehicle_upper_bound, self.config.visibility_compare_max_distance),
                )
            )
        return self.vehicle_upper_bound

    def _compute_fused_visibility(
        self,
        visibility_compare: float,
        visibility_trans: Optional[float],
        visibility_detect: Optional[float],
    ) -> float:
        if self._suppress_fusion_update:
            return max(float(visibility_compare), self.config.visibility_min_distance)

        min_distance = max(self.config.visibility_min_distance, 0.0)
        compare_value = max(float(visibility_compare), min_distance)
        alpha_compare = clamp(self.config.visibility_fusion_alpha_compare, 0.0, 1.0)
        if self.smoothed_compare is None:
            self.smoothed_compare = compare_value
        else:
            self.smoothed_compare = (1.0 - alpha_compare) * self.smoothed_compare + alpha_compare * compare_value

        trans_input: Optional[float] = None
        if visibility_trans is not None:
            trans_value = max(float(visibility_trans), min_distance)
            alpha_trans = clamp(self.config.visibility_fusion_alpha_trans, 0.0, 1.0)
            if self.smoothed_trans is None:
                self.smoothed_trans = trans_value
            else:
                self.smoothed_trans = (1.0 - alpha_trans) * self.smoothed_trans + alpha_trans * trans_value
        if self.smoothed_trans is not None:
            trans_input = self.smoothed_trans

        weight_compare = max(self.config.visibility_compare_weight, 0.0)
        weight_trans = max(self.config.visibility_trans_weight, 0.0)
        fused_raw = self.smoothed_compare if self.smoothed_compare is not None else compare_value
        if trans_input is not None and weight_trans > 0.0:
            total_weight = weight_compare + weight_trans
            if total_weight > 1e-6:
                fused_raw = (weight_compare * self.smoothed_compare + weight_trans * trans_input) / total_weight
            else:
                fused_raw = trans_input
        elif weight_compare <= 0.0 and trans_input is not None:
            fused_raw = trans_input

        fused_raw = max(fused_raw, min_distance)
        fused_raw = min(fused_raw, self.config.visibility_compare_max_distance)

        upper_bound = self._update_vehicle_upper_bound(visibility_detect)
        if upper_bound is not None:
            fused_raw = min(fused_raw, upper_bound)
            fused_raw = max(fused_raw, min_distance)

        alpha_final = clamp(self.config.visibility_fusion_alpha_final, 0.0, 1.0)
        if self.fused_visibility is None:
            self.fused_visibility = fused_raw
        else:
            self.fused_visibility = (1.0 - alpha_final) * self.fused_visibility + alpha_final * fused_raw

        self.fused_visibility = float(
            max(min_distance, min(self.fused_visibility, self.config.visibility_compare_max_distance))
        )
        return self.fused_visibility

    def _update_vehicle_history(self, vehicle_boxes: Sequence[BoundingBox], frame_width: int) -> None:
        if self.vehicle_column_history is None or self.vehicle_column_history.shape[0] != frame_width:
            self.vehicle_column_history = np.zeros(frame_width, dtype=np.float32)
        decay = self.config.vehicle_roi_history_decay
        self.vehicle_column_history *= decay
        for box in vehicle_boxes:
            start = int(clamp(box.x - self.config.vehicle_roi_expand_margin, 0, frame_width - 1))
            end = int(clamp(box.x + box.width + self.config.vehicle_roi_expand_margin, 0, frame_width))
            if end <= start:
                continue
            self.vehicle_column_history[start:end] += 1.0

    def _expand_roi_with_vehicle_history(
        self,
        roi_mask: Optional[np.ndarray],
        vanish_point_row: float,
        frame_shape: Tuple[int, int],
    ) -> Optional[np.ndarray]:
        if roi_mask is None or self.vehicle_column_history is None:
            return roi_mask
        active = self.vehicle_column_history >= self.config.vehicle_roi_column_threshold
        if not np.any(active):
            return roi_mask
        height, width = frame_shape
        top = int(max(0, vanish_point_row - self.config.vehicle_roi_expand_margin))
        bottom = int(min(height - 1, vanish_point_row + self.config.vehicle_roi_expand_rows))
        if top >= bottom:
            return roi_mask
        expanded = roi_mask.copy()
        margin = self.config.vehicle_roi_expand_margin
        active_indices = np.where(active)[0]
        if active_indices.size == 0:
            return roi_mask
        run_start = active_indices[0]
        previous = run_start
        for col in active_indices[1:]:
            if col != previous + 1:
                start = int(max(0, run_start - margin))
                end = int(min(width - 1, previous + margin))
                cv2.rectangle(expanded, (start, top), (end, bottom), 255, thickness=-1)
                run_start = col
            previous = col
        start = int(max(0, run_start - margin))
        end = int(min(width - 1, previous + margin))
        cv2.rectangle(expanded, (start, top), (end, bottom), 255, thickness=-1)
        return expanded

    def _refine_vanish_with_segments(
        self,
        segments: Sequence[LaneSegment],
        vanish_point_row: float,
        frame_width: int,
    ) -> None:
        if not segments:
            return

        candidates: List[float] = []
        for seg in segments:
            contour = seg.contour
            if contour.shape[0] < 2:
                continue
            p0 = contour[0]
            p1 = contour[-1]
            if p0[1] == p1[1]:
                continue
            if p0[1] < p1[1]:
                x_top, y_top = p0
                x_bottom, y_bottom = p1
            else:
                x_top, y_top = p1
                x_bottom, y_bottom = p0
            dy = y_bottom - y_top
            if abs(dy) < 1e-3:
                continue
            dx = x_bottom - x_top
            ratio = (vanish_point_row - y_top) / dy
            x_at_vanish = x_top + ratio * dx
            if np.isfinite(x_at_vanish) and -frame_width * 0.5 <= x_at_vanish <= frame_width * 1.5:
                candidates.append(float(x_at_vanish))

        if not candidates:
            return

        refined_x = float(np.median(candidates))
        refined_x = max(0.0, min(float(frame_width - 1), refined_x))
        self.locked_vanish_point = (refined_x, vanish_point_row)
        self.vanish_outlier_count = 0

    def _update_vanish_point_lock(self, cand_x: float, cand_y: float) -> None:
        if self.locked_vanish_point is None:
            self.locked_vanish_point = (cand_x, cand_y)
            self.vanish_outlier_count = 0
            return

        locked_x, locked_y = self.locked_vanish_point
        tol = self.config.vanish_tolerance_px
        if abs(cand_x - locked_x) <= tol and abs(cand_y - locked_y) <= tol:
            alpha = self.config.vanish_lock_alpha
            self.locked_vanish_point = (
                (1 - alpha) * locked_x + alpha * cand_x,
                (1 - alpha) * locked_y + alpha * cand_y,
            )
            self.vanish_outlier_count = 0
        else:
            self.vanish_outlier_count += 1
            if self.vanish_outlier_count >= self.config.vanish_relock_frames:
                self.locked_vanish_point = (cand_x, cand_y)
                self.vanish_outlier_count = 0
