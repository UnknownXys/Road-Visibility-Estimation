from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from road_visibility.config import PipelineConfig
from road_visibility.visibility import RoadVisibilityEstimator
from road_visibility.types import VisibilityEstimate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process the first segment of a video and save detection overlays.")
    parser.add_argument("--video", required=True, help="Path to the input video.")
    parser.add_argument("--output-dir", required=True, help="Directory to store clear reference and overlays.")
    parser.add_argument("--duration", type=float, default=300.0, help="Duration (seconds) to process from the start.")
    parser.add_argument("--sample-stride", type=int, default=90, help="Save an overlay every N frames after warm-up.")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def draw_overlay(frame: np.ndarray, estimate: VisibilityEstimate) -> np.ndarray:
    overlay = frame.copy()
    for seg in estimate.lane_segments:
        contour = seg.contour.astype(np.int32)
        if contour.shape[0] >= 2:
            cv2.polylines(overlay, [contour], isClosed=False, color=(0, 255, 0), thickness=2)
    vp_row = int(round(estimate.vanish_point_row))
    vp_col = int(round(estimate.vanish_point_col))
    cv2.drawMarker(
        overlay,
        (vp_col, vp_row),
        (0, 0, 255),
        markerType=cv2.MARKER_CROSS,
        markerSize=18,
        thickness=2,
    )
    text = f"vp=({estimate.vanish_point_col:.1f},{estimate.vanish_point_row:.1f}) lanes={len(estimate.lane_segments)}"
    cv2.putText(
        overlay,
        text,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return overlay


def compute_clear_frame(frames: List[np.ndarray]) -> np.ndarray:
    if not frames:
        raise ValueError("No frames available to compute clear frame.")
    accum = np.zeros_like(frames[0], dtype=np.float64)
    for frame in frames:
        accum += frame.astype(np.float64)
    averaged = (accum / len(frames)).astype(np.uint8)
    return averaged


def process_video_segment(
    video_path: Path,
    output_dir: Path,
    duration: float,
    sample_stride: int,
) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Unable to open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 1e-3:
        fps = 30.0
    max_frames = int(duration * fps)

    warmup_frames = max(int(fps * 5), 1)
    warmup_frames = min(warmup_frames, max_frames, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or warmup_frames)

    collected: List[np.ndarray] = []
    frame_count = 0
    while frame_count < warmup_frames:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        collected.append(frame)
        frame_count += 1
    if not collected:
        cap.release()
        raise ValueError("Video contains no frames for warm-up.")

    clear_frame = compute_clear_frame(collected)
    ensure_dir(output_dir)
    clear_path = output_dir / "clear_reference.png"
    cv2.imwrite(str(clear_path), clear_frame)

    config = PipelineConfig()
    estimator = RoadVisibilityEstimator(config=config)
    estimate = estimator.initialize(clear_frame)
    overlay = draw_overlay(clear_frame, estimate)
    overlay_path = output_dir / "overlay_clear_reference.png"
    cv2.imwrite(str(overlay_path), overlay)
    cap.release()


def main() -> None:
    args = parse_args()
    video_path = Path(args.video)
    output_dir = Path(args.output_dir)
    process_video_segment(
        video_path=video_path,
        output_dir=output_dir,
        duration=args.duration,
        sample_stride=max(1, args.sample_stride),
    )
    print(f"[info] saved outputs under {output_dir}")


if __name__ == "__main__":
    main()
