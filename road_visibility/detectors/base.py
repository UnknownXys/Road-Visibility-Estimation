from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from ..config import PipelineConfig
from ..types import LaneSegment


@dataclass
class BaseLaneDetector:
    config: PipelineConfig

    def detect(
        self,
        frame: np.ndarray,
        vanish_point_row: float,
        vanish_point_col: Optional[float] = None,
        roi_mask: Optional[np.ndarray] = None,
    ) -> Sequence[LaneSegment]:
        raise NotImplementedError
