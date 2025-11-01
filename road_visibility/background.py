from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .config import PipelineConfig
from .utils import dilate_erode, ensure_directory


@dataclass
class BackgroundEdgeModel:
    config: PipelineConfig
    accumulator: Optional[np.ndarray] = None
    initialized: bool = False

    def update(self, edge_frame: np.ndarray) -> None:
        if self.accumulator is None or self.accumulator.shape != edge_frame.shape:
            self.accumulator = edge_frame.astype(np.float32)
            self.initialized = True
            return

        cv2.accumulateWeighted(
            edge_frame.astype(np.float32),
            self.accumulator,
            self.config.background_alpha,
        )
        self.initialized = True

    def get_reference(self) -> Optional[np.ndarray]:
        if not self.initialized or self.accumulator is None:
            return None
        reference = np.clip(self.accumulator, 0, 255).astype(np.uint8)
        mask = reference < 10
        reference[mask] = 0
        return dilate_erode(reference, self.config.morph_kernel_size)

    def save(self, path: str) -> None:
        if not self.initialized:
            return
        ensure_directory(path)
        reference = self.get_reference()
        if reference is None:
            return
        cv2.imwrite(path, reference)
