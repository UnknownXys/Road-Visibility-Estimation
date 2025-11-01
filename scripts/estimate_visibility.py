import argparse
from pathlib import Path

import cv2

from road_visibility.config import PipelineConfig
from road_visibility.detectors import (
    LazyUltralyticsVehicleDetector,
    OnnxLaneSegmentationModel,
)
from road_visibility.visibility import RoadVisibilityEstimator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate road visibility from images.")
    parser.add_argument("--clear", required=True, help="Path to a clear-day reference frame.")
    parser.add_argument("--fog", required=True, help="Path to a foggy frame.")
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
            except ValueError as exc:  # pragma: no cover
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

    clear_frame = cv2.imread(args.clear)
    if clear_frame is None:
        raise FileNotFoundError(f"Failed to read clear frame: {args.clear}")
    fog_frame = cv2.imread(args.fog)
    if fog_frame is None:
        raise FileNotFoundError(f"Failed to read fog frame: {args.fog}")

    print("[info] Initializing with clear frame…")
    estimate_clear = estimator.initialize(clear_frame)
    print(
        f"  Vanish point: ({estimate_clear.vanish_point_col:.1f}, {estimate_clear.vanish_point_row:.1f})"
    )
    print(f"  Lambda (lane): {estimate_clear.lambda_from_lane}")

    print("[info] Estimating visibility on foggy frame…")
    estimate_fog = estimator.estimate(fog_frame)
    print(f"  Lambda fused: {estimate_fog.lambda_fused:.1f}")
    print(f"  Visibility (edge compare): {estimate_fog.visibility_compare:.1f} m")
    print(f"  Visibility (vehicle detect): {estimate_fog.visibility_detect:.1f} m")


if __name__ == "__main__":
    main()
