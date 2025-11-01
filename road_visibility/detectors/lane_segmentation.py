from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

try:
    import onnxruntime as ort  # type: ignore
except ImportError:  # pragma: no cover
    ort = None

from ..config import PipelineConfig


@dataclass
class BaseLaneSegmentationModel:
    config: PipelineConfig

    def infer(self, frame: np.ndarray) -> Optional[np.ndarray]:
        return None


@dataclass
class OnnxLaneSegmentationModel(BaseLaneSegmentationModel):
    weights_path: str
    input_size: Optional[tuple[int, int]] = None
    providers: Optional[list[str]] = None
    input_name: Optional[str] = None
    output_name: Optional[str] = None
    _session: Optional["ort.InferenceSession"] = None

    def infer(self, frame: np.ndarray) -> Optional[np.ndarray]:
        if ort is None:
            print("[OnnxLaneSegmentationModel] onnxruntime not available; skipping segmentation.")
            return None
        if self._session is None:
            self._load_session()
        if self._session is None:
            return None

        img = frame
        if self.input_size:
            img = cv2.resize(img, self.input_size[::-1])
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        blob = img_rgb.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[None, ...]

        inputs = {self.input_name or self._session.get_inputs()[0].name: blob}
        outputs = self._session.run([self.output_name] if self.output_name else None, inputs)
        output = outputs[0]
        if output.ndim == 4:
            output = output[0]
        if output.ndim == 3:
            output = np.argmax(output, axis=0)
        elif output.ndim == 2:
            output = (output > self.config.segmentation_threshold).astype(np.uint8)
        else:
            return None

        mask = output.astype(np.uint8)
        if self.input_size:
            mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0).astype(np.uint8) * 255
        return mask

    def _load_session(self) -> None:
        try:
            providers = self.providers or ["CUDAExecutionProvider", "CPUExecutionProvider"]
            self._session = ort.InferenceSession(self.weights_path, providers=providers)
        except Exception as exc:  # pragma: no cover
            print(f"[OnnxLaneSegmentationModel] failed to load model: {exc}")
            self._session = None
