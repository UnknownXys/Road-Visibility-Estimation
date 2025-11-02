from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from road_visibility.config import PipelineConfig
from road_visibility.utils import ensure_directory
from road_visibility.visibility import RoadVisibilityEstimator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Debug visibility compare/transmittance values for a single frame."
    )
    parser.add_argument("--clear", required=True, help="Path to the clear reference image.")
    parser.add_argument("--image", required=True, help="Path to the foggy/target image.")
    parser.add_argument(
        "--output",
        help="Optional path for the annotated image. Defaults to <image>_visibility_debug.png",
    )
    parser.add_argument(
        "--save-trans-map",
        help="Optional path to save the raw transmittance map for inspection.",
    )
    return parser.parse_args()


def annotate_rows(
    frame: np.ndarray,
    compare_row: Optional[float],
    trans_row: Optional[float],
    compare_value: float,
    trans_value: Optional[float],
    fused_value: Optional[float],
    detect_value: Optional[float],
) -> np.ndarray:
    annotated = frame.copy()
    height, width = annotated.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    text_lines = [
        f"vis_compare = {compare_value:7.1f} m",
    ]
    if trans_value is not None:
        text_lines.append(f"vis_trans   = {trans_value:7.1f} m")
    else:
        text_lines.append("vis_trans   =   n/a")
    if fused_value is not None:
        text_lines.append(f"vis_fused   = {fused_value:7.1f} m")
    else:
        text_lines.append("vis_fused   =   n/a")
    if detect_value is not None:
        text_lines.append(f"vis_detect  = {detect_value:7.1f} m")
    else:
        text_lines.append("vis_detect  =   n/a")

    for idx, text in enumerate(text_lines):
        org = (10, 28 + idx * 24)
        cv2.putText(annotated, text, org, font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    if compare_row is not None:
        y = int(round(compare_row))
        y = max(0, min(height - 1, y))
        cv2.line(annotated, (0, y), (width - 1, y), (0, 165, 255), 2)
        label = f"compare row {compare_row:.1f}"
        label_pos = (10, min(height - 10, y + 24))
        cv2.putText(annotated, label, label_pos, font, 0.6, (0, 165, 255), 2, cv2.LINE_AA)

    if trans_row is not None:
        y = int(round(trans_row))
        y = max(0, min(height - 1, y))
        cv2.line(annotated, (0, y), (width - 1, y), (0, 255, 0), 2)
        label = f"trans row {trans_row:.1f}"
        label_pos = (10, max(20, y - 10))
        cv2.putText(annotated, label, label_pos, font, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

    return annotated


def main() -> None:
    args = parse_args()

    clear_path = Path(args.clear)
    image_path = Path(args.image)
    output_path = (
        Path(args.output)
        if args.output
        else image_path.with_name(f"{image_path.stem}_visibility_debug.png")
    )

    clear_frame = cv2.imread(str(clear_path))
    if clear_frame is None:
        raise FileNotFoundError(f"Failed to read clear reference image: {clear_path}")

    target_frame = cv2.imread(str(image_path))
    if target_frame is None:
        raise FileNotFoundError(f"Failed to read target image: {image_path}")

    config = PipelineConfig()
    estimator = RoadVisibilityEstimator(config=config)

    estimator.initialize(clear_frame)

    processed_frame = estimator._resize_frame(target_frame)
    estimate = estimator.estimate(processed_frame)

    trans_map: Optional[np.ndarray] = None
    roi_mask = estimate.roi_mask
    if args.save_trans_map:
        if estimator.reference_dark_channel is None:
            raise RuntimeError("Transmittance map requested but estimator has no reference dark channel.")
        trans_map = estimator._compute_transmittance_map(processed_frame)
        if roi_mask is not None and roi_mask.shape == trans_map.shape:
            if roi_mask.dtype != np.uint8:
                mask = roi_mask > 0
            else:
                mask = roi_mask.astype(bool)
            masked = np.zeros_like(trans_map)
            masked[mask] = trans_map[mask]
            trans_map = masked

    compare_row = estimator.last_compare_row
    trans_row = estimator.last_trans_row

    print("[info] visibility results:")
    print(f"  vis_compare = {estimate.visibility_compare:.1f} m (row={compare_row if compare_row is not None else 'n/a'})")
    if estimate.visibility_transmittance is not None:
        trans_row_display = f"{trans_row:.1f}" if trans_row is not None else "n/a"
        print(
            f"  vis_trans   = {estimate.visibility_transmittance:.1f} m "
            f"(row={trans_row_display})"
        )
    else:
        print("  vis_trans   = n/a (insufficient data)")
    if estimate.visibility_fused is not None:
        print(f"  vis_fused   = {estimate.visibility_fused:.1f} m")
    else:
        print("  vis_fused   = n/a")
    if estimate.visibility_detect is not None:
        print(f"  vis_detect  = {estimate.visibility_detect:.1f} m")
    else:
        print("  vis_detect  = n/a")
    if estimate.transmittance_percentile_value is not None:
        print(f"  tau_percentile = {estimate.transmittance_percentile_value:.3f}")
    if estimate.mean_transmittance is not None:
        print(f"  mean_trans      = {estimate.mean_transmittance:.3f}")

    annotated = annotate_rows(
        processed_frame,
        compare_row,
        trans_row,
        estimate.visibility_compare,
        estimate.visibility_transmittance,
        estimate.visibility_fused,
        estimate.visibility_detect,
    )

    if annotated.shape[:2] != target_frame.shape[:2]:
        annotated = cv2.resize(annotated, (target_frame.shape[1], target_frame.shape[0]), interpolation=cv2.INTER_LINEAR)

    ensure_directory(str(output_path.parent))
    cv2.imwrite(str(output_path), annotated)
    print(f"[info] annotated image saved to {output_path}")

    if args.save_trans_map:
        if trans_map is None:
            print("[warn] transmittance map not generated.")
        else:
            trans_output_path = Path(args.save_trans_map)
            ensure_directory(str(trans_output_path.parent))
            trans_norm = np.clip(trans_map * 255.0, 0, 255).astype(np.uint8)
            if trans_norm.shape[:2] != target_frame.shape[:2]:
                trans_norm = cv2.resize(
                    trans_norm,
                    (target_frame.shape[1], target_frame.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
            cv2.imwrite(str(trans_output_path), trans_norm)
            print(f"[info] transmittance map saved to {trans_output_path}")


if __name__ == "__main__":
    main()
