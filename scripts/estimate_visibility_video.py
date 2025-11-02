import argparse

from road_visibility.config import PipelineConfig
from road_visibility.detectors import LazyUltralyticsVehicleDetector
from road_visibility.types import FrameVisibility
from road_visibility.visibility import RoadVisibilityEstimator
from road_visibility.video import VideoVisibilityProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate road visibility for a video stream.")
    parser.add_argument("--video", required=True, help="Path to the input video.")
    parser.add_argument(
        "--clear-image",
        help="Optional clear-scene image; otherwise the first 60 seconds of the video are averaged.",
    )
    parser.add_argument(
        "--warmup-fraction",
        type=float,
        default=0.0,
        help="Fraction of frames used for warm-up (<=0 uses the first 60 seconds).",
    )
    parser.add_argument("--frame-stride", type=int, default=1, help="Process every Nth frame after warm-up.")
    parser.add_argument("--use-yolo", action="store_true", help="Enable YOLO-based vehicle detection.")
    parser.add_argument("--yolo-weights", default="checkpoints/yolov8n.pt", help="Weights file for Ultralytics YOLO.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PipelineConfig()

    vehicle_detectors = []
    if args.use_yolo:
        vehicle_detectors.append(LazyUltralyticsVehicleDetector(config, model_name=args.yolo_weights))

    estimator = RoadVisibilityEstimator(
        config=config,
        vehicle_detectors=vehicle_detectors,
    )

    processor = VideoVisibilityProcessor(
        estimator=estimator,
        warmup_fraction=args.warmup_fraction,
        frame_stride=args.frame_stride,
    )

    def _print_progress(entry: FrameVisibility) -> None:
        est = entry.estimate
        parts = [
            f"[frame {entry.frame_index:05d}]",
            f"t={entry.timestamp:7.2f}s",
            f"vis_compare={est.visibility_compare:7.1f}m",
        ]
        if est.visibility_transmittance is not None:
            parts.append(f"vis_trans={est.visibility_transmittance:7.1f}m")
        if est.visibility_fused is not None:
            parts.append(f"fused={est.visibility_fused:7.1f}m")
        print(" ".join(parts), flush=True)

    results = processor.process_video(
        args.video,
        clear_image_path=args.clear_image,
        progress_hook=_print_progress,
    )

    print(f"[info] processed {len(results)} frames after warm-up.")
    if results:
        last = results[-1]
        print(f"  Last frame index: {last.frame_index}")
        print(f"  Last visibility (edge compare): {last.estimate.visibility_compare:.1f} m")
        if last.estimate.visibility_transmittance is not None:
            print(f"  Last visibility (transmittance): {last.estimate.visibility_transmittance:.1f} m")
        if last.estimate.visibility_fused is not None:
            print(f"  Last visibility (fused): {last.estimate.visibility_fused:.1f} m")


if __name__ == "__main__":
    main()


