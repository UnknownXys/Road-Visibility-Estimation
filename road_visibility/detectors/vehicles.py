from __future__ import annotations

from dataclasses import dataclass
import math
from typing import ClassVar, List, Optional

import numpy as np

from ..config import PipelineConfig
from ..types import BoundingBox


@dataclass
class BaseVehicleDetector:
    """
    Interface for pluggable vehicle detectors. The default implementation returns no detections.
    """

    config: PipelineConfig

    def detect(self, frame: np.ndarray) -> List[BoundingBox]:
        return []


@dataclass
class LazyUltralyticsVehicleDetector(BaseVehicleDetector):
    model_name: str = "yolov8n.pt"
    confidence: float = 0.25
    classes: Optional[List[int]] = None  # defaults to car/bus/truck automatically
    device: Optional[str] = None

    _model: Optional[object] = None
    _ultralytics_import_attempted: ClassVar[bool] = False
    _ultralytics_available: ClassVar[bool] = True
    _yolo_ctor: ClassVar[Optional[object]] = None

    def detect(self, frame: np.ndarray) -> List[BoundingBox]:
        if self._model is None:
            self._load()
        if self._model is None:
            return []

        height, width = frame.shape[:2]
        stride = 32
        target_height = max(stride, int(math.ceil(height / stride) * stride))
        target_width = max(stride, int(math.ceil(width / stride) * stride))
        results = self._model(
            frame,
            imgsz=(target_height, target_width),
            conf=self.confidence,
            device=self.device,
            verbose=False,
        )
        boxes: List[BoundingBox] = []
        for res in results:
            if not hasattr(res, "boxes"):
                continue
            for b in res.boxes:
                cls_id = int(b.cls)
                if self.classes and cls_id not in self.classes:
                    continue
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                width = max(1, x2 - x1)
                height = max(1, y2 - y1)
                boxes.append(
                    BoundingBox(
                        x=x1,
                        y=y1,
                        width=width,
                        height=height,
                        confidence=float(b.conf),
                        label="vehicle",
                    )
                )
        return boxes

    def _load(self) -> None:
        cls = type(self)

        if cls._ultralytics_import_attempted and not cls._ultralytics_available:
            return

        if not cls._ultralytics_import_attempted:
            try:
                from ultralytics import YOLO  # type: ignore
            except ImportError:
                print(
                    "[LazyUltralyticsVehicleDetector] ultralytics not installed; "
                    "skipping vehicle detections."
                )
                cls._ultralytics_available = False
                cls._ultralytics_import_attempted = True
                self._model = None
                return
            cls._yolo_ctor = YOLO
            cls._ultralytics_available = True
            cls._ultralytics_import_attempted = True
        elif cls._ultralytics_available and cls._yolo_ctor is None:
            # Import succeeded previously but constructor was not cached (different process instance).
            from ultralytics import YOLO  # type: ignore
            cls._yolo_ctor = YOLO

        if not cls._ultralytics_available or cls._yolo_ctor is None:
            self._model = None
            return

        if self.classes is None:
            # Default class ID for COCO car = 2
            self.classes = [2]
        self._model = cls._yolo_ctor(self.model_name)
