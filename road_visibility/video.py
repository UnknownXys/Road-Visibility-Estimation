from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import cv2
import numpy as np
from pathlib import Path

from .types import FrameVisibility
from .visibility import RoadVisibilityEstimator
from .utils import ensure_directory


@dataclass
class VideoVisibilityProcessor:
    estimator: RoadVisibilityEstimator
    warmup_fraction: float = 0.2
    frame_stride: int = 1
    save_clear_scene: bool = True
    _last_clear_scene_ts: float = field(default=float("-inf"), init=False, repr=False)

    def __post_init__(self) -> None:
        self.estimator.use_transmittance_video_fusion = True

    def _resize_for_estimator(self, frame: np.ndarray) -> np.ndarray:
        target_w = max(int(self.estimator.config.frame_target_width), 0)
        target_h = max(int(self.estimator.config.frame_target_height), 0)
        if target_w > 0 and target_h > 0 and (frame.shape[1] != target_w or frame.shape[0] != target_h):
            return cv2.resize(frame, (target_w, target_h))
        return frame

    def process_video(
        self,
        video_path: str,
        clear_image_path: Optional[str] = None,
        progress_hook: Optional[Callable[[FrameVisibility], None]] = None,
    ) -> List[FrameVisibility]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Unable to open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(fps) or fps <= 1e-3:
            fps = 30.0

        warmup_frames: List[np.ndarray] = []
        accum: Optional[np.ndarray] = None
        warmup_count = 0

        if clear_image_path is None:
            warmup_count = int(total_frames * self.warmup_fraction) if total_frames > 0 else 0
            warmup_count = max(warmup_count, 1)

            for _ in range(warmup_count):
                ret, frame = cap.read()
                if not ret:
                    break
                frame = self._resize_for_estimator(frame)
                warmup_frames.append(frame)
                frame_float = frame.astype(np.float32)
                if accum is None:
                    accum = frame_float
                else:
                    accum += frame_float

        if clear_image_path:
            clear_frame = cv2.imread(clear_image_path)
            if clear_frame is None:
                cap.release()
                raise FileNotFoundError(f"Failed to read clear image: {clear_image_path}")
            clear_frame = self._resize_for_estimator(clear_frame)
            self.estimator.initialize(clear_frame)
        else:
            if not warmup_frames:
                cap.release()
                raise ValueError("Video contains no frames for warm-up.")
            averaged = (accum / max(len(warmup_frames), 1)).astype(np.uint8)
            self.estimator.warmup(averaged, additional_frames=warmup_frames)

        results: List[FrameVisibility] = []
        frames_consumed = len(warmup_frames)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = self._resize_for_estimator(frame)
            frames_consumed += 1
            if self.frame_stride > 1 and ((frames_consumed - len(warmup_frames)) % self.frame_stride != 0):
                continue
            estimate = self.estimator.estimate(frame)
            frame_index = frames_consumed - 1
            timestamp = frame_index / fps
            if (
                self.save_clear_scene
                and self.estimator.is_clear_scene(estimate.mean_transmittance)
                and timestamp - self._last_clear_scene_ts >= self.estimator.config.clear_scene_min_gap_seconds
            ):
                self._save_clear_scene_frame(frame, frame_index, timestamp, estimate.mean_transmittance or 0.0)
                self._last_clear_scene_ts = timestamp
            record = FrameVisibility(
                frame_index=frame_index,
                timestamp=timestamp,
                estimate=estimate,
            )
            if progress_hook is not None:
                progress_hook(record)
            results.append(record)

        cap.release()
        return results

    def _save_clear_scene_frame(
        self,
        frame: np.ndarray,
        frame_index: int,
        timestamp: float,
        mean_transmittance: float,
    ) -> None:
        output_dir = self.estimator.config.clear_scene_save_dir
        ensure_directory(output_dir)
        filename = f"clear_{frame_index:06d}_t{timestamp:.2f}_trans{mean_transmittance:.2f}.png"
        path = Path(output_dir) / filename
        cv2.imwrite(str(path), frame)
