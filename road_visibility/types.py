from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class BoundingBox:
    x: int
    y: int
    width: int
    height: int
    confidence: float = 1.0
    label: str = "vehicle"

    def bottom(self) -> int:
        return self.y + self.height

    def top(self) -> int:
        return self.y

    def center(self) -> Tuple[float, float]:
        return self.x + self.width * 0.5, self.y + self.height * 0.5


@dataclass
class LaneSegment:
    contour: np.ndarray
    bounding_box: BoundingBox
    length_px: float


@dataclass
class VisibilityEstimate:
    vanish_point_row: float
    vanish_point_col: float
    lambda_from_lane: Optional[float]
    lambda_from_vehicle: Optional[float]
    lambda_fused: float
    visibility_detect: float
    visibility_compare: float
    roi_mask: Optional[np.ndarray] = None
    lane_segments: List[LaneSegment] = field(default_factory=list)
    vehicle_boxes: List[BoundingBox] = field(default_factory=list)
    mean_transmittance: Optional[float] = None
    transmittance_histogram: Optional[np.ndarray] = None
    transmittance_bins: Optional[np.ndarray] = None
    visibility_transmittance: Optional[float] = None
    transmittance_percentile_value: Optional[float] = None


@dataclass
class FrameVisibility:
    frame_index: int
    timestamp: float
    estimate: VisibilityEstimate
