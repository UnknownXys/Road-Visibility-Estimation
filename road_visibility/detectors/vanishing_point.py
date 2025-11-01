from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from ..config import PipelineConfig
from ..utils import (
    compute_intersections,
    median_point,
    read_gray,
)


@dataclass
class VanishingPointDetector:
    config: PipelineConfig
    _last_point: Optional[Tuple[float, float]] = None

    def detect(self, frame: np.ndarray) -> Tuple[float, float]:
        gray = read_gray(frame)
        edges = cv2.Canny(gray, *self.config.canny_thresholds)

        lines = cv2.HoughLines(edges, 1, np.pi / 180, 120)
        if lines is None or len(lines) < 2:
            return self._fallback(frame)

        filtered = []
        for rho, theta in lines[:, 0]:
            angle_deg = abs(theta * 180 / math.pi - 90)
            if angle_deg < 15:  # near-vertical
                filtered.append((rho, theta))
            elif angle_deg > 70:  # near-horizontal lines add noise
                continue
            else:
                filtered.append((rho, theta))

        if len(filtered) < 2:
            return self._fallback(frame)

        intersections = compute_intersections(filtered[:40])
        if len(intersections) < 5:
            return self._fallback(frame)

        vp = median_point(intersections)
        if not np.isfinite(vp[1]) or vp[1] < 0:
            return self._fallback(frame)

        self._last_point = vp
        return vp

    def _fallback(self, frame: np.ndarray) -> Tuple[float, float]:
        if self._last_point is not None:
            return self._last_point
        h, w = frame.shape[:2]
        return w * 0.5, h * 0.45
