from __future__ import annotations

import argparse

import cv2

from road_visibility.config import PipelineConfig
from road_visibility.detectors import LazyUltralyticsVehicleDetector
from road_visibility.visibility import RoadVisibilityEstimator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate road visibility from images.")
    parser.add_argument("--clear", required=True, help="Path to a clear-day reference frame.")
    parser.add_argument("--fog", required=True, help="Path to a foggy frame.")
    parser.add_argument("--use-yolo", action="store_true", help="Enable YOLO-based vehicle detection.")
    parser.add_argument("--yolo-weights", default="yolov8n.pt", help="Weights file for Ultralytics YOLO.")
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

    clear_frame = cv2.imread(args.clear)
    if clear_frame is None:
        raise FileNotFoundError(f"Failed to read clear frame: {args.clear}")
    fog_frame = cv2.imread(args.fog)
    if fog_frame is None:
        raise FileNotFoundError(f"Failed to read fog frame: {args.fog}")

    print("[info] Initializing with clear frame...")
    estimate_clear = estimator.initialize(clear_frame)
    print(
        f"  Vanish point: ({estimate_clear.vanish_point_col:.1f}, {estimate_clear.vanish_point_row:.1f})"
    )

    print("[info] Estimating visibility on foggy frame...")
    estimate_fog = estimator.estimate(fog_frame)
    print(f"  Visibility (edge compare): {estimate_fog.visibility_compare:.1f} m")
    if estimate_fog.visibility_transmittance is not None:
        print(f"  Visibility (transmittance): {estimate_fog.visibility_transmittance:.1f} m")
    if estimate_fog.visibility_fused is not None:
        print(f"  Visibility (fused): {estimate_fog.visibility_fused:.1f} m")


if __name__ == "__main__":
    main()
