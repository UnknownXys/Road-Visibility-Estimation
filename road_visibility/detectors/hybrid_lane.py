from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import cv2
import numpy as np

from ..config import PipelineConfig
from ..types import BoundingBox, LaneSegment
from ..utils import clamp
from .base import BaseLaneDetector
from .lane_markings import LaneDashDetector
from .lane_segmentation import BaseLaneSegmentationModel


@dataclass
class HybridLaneDetector(BaseLaneDetector):
    segmentation_model: Optional[BaseLaneSegmentationModel] = None
    fallback_detector: Optional[LaneDashDetector] = None
    _kernel: Optional[np.ndarray] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.fallback_detector is None:
            self.fallback_detector = LaneDashDetector(self.config)

    def detect(
        self,
        frame: np.ndarray,
        vanish_point_row: float,
        vanish_point_col: Optional[float] = None,
        roi_mask: Optional[np.ndarray] = None,
    ) -> Sequence[LaneSegment]:
        segments: List[LaneSegment] = []

        if self.segmentation_model is not None:
            mask = self.segmentation_model.infer(frame)
            if mask is not None:
                if roi_mask is not None and roi_mask.shape[:2] == mask.shape[:2]:
                    mask = cv2.bitwise_and(mask, roi_mask)
                segments.extend(self._segments_from_mask(mask, vanish_point_row))

        fallback_segments = list(
            self.fallback_detector.detect(
                frame,
                vanish_point_row,
                vanish_point_col=vanish_point_col,
                roi_mask=roi_mask,
            )
        )

        segments = self._merge_segments(segments, fallback_segments)
        segments.sort(key=lambda seg: seg.bounding_box.bottom())
        return segments

    def _segments_from_mask(self, mask: np.ndarray, vanish_point_row: float) -> List[LaneSegment]:
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)
        if self._kernel is None:
            k = self.config.segmentation_morph_kernel
            self._kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        processed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel, iterations=1)
        processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, self._kernel, iterations=1)

        contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h, w = mask.shape[:2]
        vanish_limit = max(0, int(vanish_point_row) - self.config.lane_roi_margin_px)
        segments: List[LaneSegment] = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.config.segmentation_min_area_px:
                continue
            x, y, width, height = cv2.boundingRect(contour)
            if y + height <= vanish_limit:
                continue
            if height < self.config.lane_min_length_px // 2 or height > self.config.lane_max_length_px * 1.2:
                continue
            aspect = width / max(height, 1)
            if aspect > self.config.lane_max_aspect_ratio * 1.5:
                continue
            box = BoundingBox(x, y, width, height, confidence=1.0, label="lane_dash/hybrid")
            length_px = float(height)
            segments.append(LaneSegment(contour=contour, bounding_box=box, length_px=length_px))

        return segments

    def _merge_segments(
        self,
        primary: Sequence[LaneSegment],
        secondary: Sequence[LaneSegment],
    ) -> List[LaneSegment]:
        if not primary:
            return list(secondary)

        merged: List[LaneSegment] = list(primary)
        tolerance = self.config.lane_merge_tolerance_px

        for seg in secondary:
            if any(
                abs(seg.bounding_box.top() - existing.bounding_box.top()) < tolerance
                and abs(seg.bounding_box.x - existing.bounding_box.x) < tolerance
                for existing in merged
            ):
                continue
            merged.append(seg)

        return merged
