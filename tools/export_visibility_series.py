import argparse
import csv
from pathlib import Path
from typing import Iterable, Optional

import cv2

from road_visibility.config import PipelineConfig
from road_visibility.types import FrameVisibility
from road_visibility.visibility import RoadVisibilityEstimator
from road_visibility.video import VideoVisibilityProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export per-frame visibility estimates for one or more videos.",
    )
    parser.add_argument(
        "--video",
        action="append",
        required=True,
        help="Path to a video file. Repeat the flag to process multiple videos.",
    )
    parser.add_argument(
        "--clear-image",
        required=True,
        help="Path to a clear-scene image shared by the videos.",
    )
    parser.add_argument(
        "--output-dir",
        default="visibility_series",
        help="Directory to store per-video CSV outputs.",
    )
    parser.add_argument(
        "--warmup-fraction",
        type=float,
        default=0.2,
        help="Fraction of frames reserved for warm-up (default: 0.2).",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Process every Nth frame after warm-up (default: 1).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Optional cap on frames processed after warm-up (0 = no limit).",
    )
    return parser.parse_args()


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def export_visibility_series(
    estimator: RoadVisibilityEstimator,
    video_path: Path,
    clear_image: Path,
    warmup_fraction: float,
    frame_stride: int,
    max_frames: int,
) -> Iterable[FrameVisibility]:
    clear_frame = cv2.imread(str(clear_image))
    if clear_frame is None:
        raise FileNotFoundError(f"Failed to read clear image: {clear_image}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Unable to open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    warmup_frames = max(1, int(total_frames * warmup_fraction)) if total_frames > 0 else 1
    warmup_frames = min(warmup_frames, total_frames) if total_frames > 0 else warmup_frames

    estimator.initialize(clear_frame)
    frame_index = 0

    # Warm-up: feed frames without recording results
    while frame_index < warmup_frames:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        estimator.estimate(frame)
        frame_index += 1

    processed = 0
    stride = max(1, frame_stride)

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        frame_index += 1

        if (frame_index - warmup_frames) % stride != 0:
            estimator.estimate(frame)
            continue

        estimate = estimator.estimate(frame)
        timestamp = frame_index / fps
        yield FrameVisibility(
            frame_index=frame_index,
            timestamp=timestamp,
            estimate=estimate,
        )
        processed += 1
        if max_frames and processed >= max_frames:
            break

    cap.release()


def write_csv(output_file: Path, series: Iterable[FrameVisibility]) -> None:
    with output_file.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "frame_index",
                "timestamp_sec",
                "visibility_compare_m",
                "visibility_detect_m",
                "lambda_fused",
                "vanish_x",
                "vanish_y",
            ]
        )
        for record in series:
            est = record.estimate
            writer.writerow(
                [
                    record.frame_index,
                    f"{record.timestamp:.3f}",
                    f"{est.visibility_compare:.3f}",
                    f"{est.visibility_detect:.3f}",
                    f"{est.lambda_fused:.2f}",
                    f"{est.vanish_point_col:.2f}",
                    f"{est.vanish_point_row:.2f}",
                ]
            )


def main() -> None:
    args = parse_args()
    video_paths = [Path(p) for p in args.video]
    clear_image = Path(args.clear_image)
    output_dir = Path(args.output_dir)
    ensure_directory(output_dir)

    config = PipelineConfig()

    for video_path in video_paths:
        print(f"[info] Processing {video_path} ...")
        estimator = RoadVisibilityEstimator(config=config)
        try:
            series = list(
                export_visibility_series(
                    estimator=estimator,
                    video_path=video_path,
                    clear_image=clear_image,
                    warmup_fraction=args.warmup_fraction,
                    frame_stride=args.frame_stride,
                    max_frames=args.max_frames,
                )
            )
        except Exception as exc:
            print(f"[error] Failed on {video_path}: {exc}")
            continue

        if not series:
            print(f"[warn] No frames processed for {video_path}")
            continue

        csv_path = output_dir / f"{video_path.stem}_visibility.csv"
        write_csv(csv_path, series)
        print(f"[info] Saved {len(series)} samples to {csv_path}")


if __name__ == "__main__":
    main()
