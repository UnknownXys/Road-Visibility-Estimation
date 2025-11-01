import argparse

from road_visibility.config import PipelineConfig
from road_visibility.detectors import LazyUltralyticsVehicleDetector, OnnxLaneSegmentationModel
from road_visibility.types import FrameVisibility
from road_visibility.visibility import RoadVisibilityEstimator
from road_visibility.video import VideoVisibilityProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate road visibility for a video stream.")
    parser.add_argument("--video", required=True, help="Path to the input video.")
    parser.add_argument(
        "--clear-image",
        help="Optional clear-scene image; otherwise the first 20%% of the video is used.",
    )
    parser.add_argument("--warmup-fraction", type=float, default=0.2, help="Fraction of frames used for warm-up.")
    parser.add_argument("--frame-stride", type=int, default=1, help="Process every Nth frame after warm-up.")
    parser.add_argument("--use-yolo", action="store_true", help="Enable YOLO-based vehicle detection.")
    parser.add_argument("--yolo-weights", default="yolov8n.pt", help="Weights file for Ultralytics YOLO.")
    parser.add_argument("--lane-model", help="Path to an ONNX lane segmentation model.")
    parser.add_argument(
        "--lane-input-size",
        help="Optional WxH resolution for the lane segmentation model (e.g. 640x360).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PipelineConfig()

    lane_model = None
    if args.lane_model:
        input_size = None
        if args.lane_input_size:
            try:
                width, height = map(int, args.lane_input_size.lower().split("x"))
                input_size = (height, width)
            except ValueError as exc:
                raise ValueError("lane-input-size must be formatted as WxH, e.g. 640x360") from exc
        lane_model = OnnxLaneSegmentationModel(
            config=config,
            weights_path=args.lane_model,
            input_size=input_size,
        )

    vehicle_detectors = []
    if args.use_yolo:
        vehicle_detectors.append(LazyUltralyticsVehicleDetector(config, model_name=args.yolo_weights))

    estimator = RoadVisibilityEstimator(
        config=config,
        lane_segmentation_model=lane_model,
        vehicle_detectors=vehicle_detectors,
    )

    processor = VideoVisibilityProcessor(
        estimator=estimator,
        warmup_fraction=args.warmup_fraction,
        frame_stride=args.frame_stride,
    )

    def _print_progress(entry: FrameVisibility) -> None:
        est = entry.estimate
        trans_part = (
            f" vis_trans={est.visibility_transmittance:7.1f}m"
            if est.visibility_transmittance is not None
            else ""
        )
        mean_trans = (
            f" mean_tau={est.mean_transmittance:.3f}"
            if est.mean_transmittance is not None
            else ""
        )
        print(
            f"[frame {entry.frame_index:05d}] "
            f"t={entry.timestamp:7.2f}s "
            f"lambda={est.lambda_fused:8.1f} "
            f"vis_compare={est.visibility_compare:7.1f}m "
            f"vis_detect={est.visibility_detect:7.1f}m"
            f"{trans_part}{mean_trans}",
            flush=True,
        )

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
        print(f"  Last visibility (vehicle detect): {last.estimate.visibility_detect:.1f} m")
        if last.estimate.visibility_transmittance is not None:
            print(f"  Last visibility (transmittance): {last.estimate.visibility_transmittance:.1f} m")
        if last.estimate.mean_transmittance is not None:
            print(f"  Last mean transmittance: {last.estimate.mean_transmittance:.3f}")


if __name__ == "__main__":
    main()
