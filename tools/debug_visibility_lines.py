from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from road_visibility.camera_model import distance_from_row
from road_visibility.config import PipelineConfig
from road_visibility.utils import clamp, count_edges, dilate_erode, polygon_from_vanish_point
from road_visibility.visibility import RoadVisibilityEstimator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualise vis_compare and vis_trans decision rows.")
    parser.add_argument("--clear", required=True, help="Path to clear reference frame.")
    parser.add_argument("--frames", nargs="+", required=True, help="Paths to foggy frames.")
    parser.add_argument("--output-dir", default="debug_visibility", help="Directory to store annotated outputs.")
    parser.add_argument("--resize-clear", action="store_true", help="Resize clear frame to each fog frame before initialise.")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def compute_vis_compare_debug(
    estimator: RoadVisibilityEstimator,
    frame: np.ndarray,
    estimate,
) -> Tuple[int, float, np.ndarray, np.ndarray, np.ndarray]:
    config = estimator.config
    edges = cv2.Canny(frame, *config.canny_thresholds)
    vanish_row = estimate.vanish_point_row

    mask = estimate.roi_mask
    if mask is not None and mask.shape == edges.shape:
        if mask.dtype != np.uint8:
            mask = np.where(mask > 0, 255, 0).astype(np.uint8)
        else:
            mask = mask.copy()
    else:
        polygon = polygon_from_vanish_point(
            frame.shape[1],
            frame.shape[0],
            vanish_row,
            config.roi_top_ratio,
            config.roi_bottom_ratio,
            config.roi_expand,
        )
        mask = np.zeros_like(edges, dtype=np.uint8)
        cv2.fillConvexPoly(mask, np.array(polygon, dtype=np.int32), 255)

    edges_masked = cv2.bitwise_and(edges, mask)
    reference = estimator.background.get_reference()
    if reference is None:
        raise RuntimeError("Background reference edge map is unavailable.")
    reference_masked = cv2.bitwise_and(reference, mask)

    edges_processed = dilate_erode(edges_masked, config.morph_kernel_size)
    reference_processed = dilate_erode(reference_masked, config.morph_kernel_size)

    row_start = int(max(0, round(vanish_row)))
    first_visible = -1
    stable = 0

    for y in range(row_start, edges.shape[0] - config.edge_window_rows):
        row_range = slice(y, y + config.edge_window_rows)
        ref_slice = reference_processed[row_range, :]
        curr_slice = edges_processed[row_range, :]

        ref_edges = count_edges(ref_slice)
        if ref_edges < config.edge_min_reference_edges:
            stable = 0
            continue

        curr_edges = count_edges(curr_slice)
        retention = curr_edges / (ref_edges + 1e-3)
        if retention >= config.edge_retention_threshold:
            stable += 1
            if stable >= config.edge_required_stable:
                first_visible = y - config.edge_required_stable // 2
                break
        else:
            stable = 0

    if first_visible < 0:
        first_visible = edges.shape[0] - 1

    distance = distance_from_row(estimate.lambda_fused, vanish_row, first_visible)
    return first_visible, distance, mask, edges_processed, reference_processed


def compute_vis_trans_debug(
    estimator: RoadVisibilityEstimator,
    frame: np.ndarray,
    estimate,
) -> Tuple[Optional[int], Optional[float], Optional[np.ndarray]]:
    if estimator.reference_dark_channel is None:
        return None, None, None

    config = estimator.config
    vanish_row = estimate.vanish_point_row

    trans_map = estimator._compute_transmittance_map(frame)  # pylint: disable=protected-access

    mask = estimate.roi_mask
    if mask is not None and mask.shape == trans_map.shape:
        mask_bool = mask.astype(bool)
    else:
        polygon = polygon_from_vanish_point(
            frame.shape[1],
            frame.shape[0],
            vanish_row,
            config.roi_top_ratio,
            config.roi_bottom_ratio,
            config.roi_expand,
        )
        mask_uint8 = np.zeros_like(trans_map, dtype=np.uint8)
        cv2.fillConvexPoly(mask_uint8, np.array(polygon, dtype=np.int32), 1)
        mask_bool = mask_uint8.astype(bool)

    values = trans_map[mask_bool]
    if values.size < config.transmittance_visibility_min_pixels:
        return None, None, None

    percentile = clamp(config.transmittance_visibility_percentile, 0.0, 1.0) * 100.0
    pixel_percentile_value = float(np.percentile(values, percentile))

    row_counts = np.sum(mask_bool, axis=1)
    row_sums = np.sum(trans_map * mask_bool, axis=1)
    row_means = np.zeros_like(row_counts, dtype=np.float32)
    valid_rows = row_counts > 0
    row_means[valid_rows] = row_sums[valid_rows] / (row_counts[valid_rows] + 1e-6)

    threshold_scaled = pixel_percentile_value * clamp(config.transmittance_row_threshold_scale, 0.0, 1.0)
    row_threshold = threshold_scaled
    if np.any(valid_rows):
        row_threshold_means = float(np.percentile(row_means[valid_rows], percentile))
        row_threshold = min(row_threshold_means, threshold_scaled)
    row_threshold = float(np.clip(row_threshold, 0.0, 1.0))

    row_mask = row_means >= row_threshold
    candidate_rows = np.where(row_mask)[0]
    if candidate_rows.size == 0:
        return None, None, None

    trans_row = int(float(np.min(candidate_rows)))
    distance = estimate.visibility_transmittance
    return trans_row, distance, mask_bool.astype(np.uint8)

def draw_horizontal_line(img: np.ndarray, row: int, color: Tuple[int, int, int], label: str) -> None:
    row = int(clamp(row, 0, img.shape[0] - 1))
    cv2.line(img, (0, row), (img.shape[1] - 1, row), color, thickness=2)
    cv2.putText(
        img,
        label,
        (10, max(30, row - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        lineType=cv2.LINE_AA,
    )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    clear_img = cv2.imread(args.clear)
    if clear_img is None:
        raise FileNotFoundError(f"Failed to read clear frame: {args.clear}")

    for frame_path in args.frames:
        frame = cv2.imread(frame_path)
        if frame is None:
            raise FileNotFoundError(f"Failed to read frame: {frame_path}")

        config = PipelineConfig()
        estimator = RoadVisibilityEstimator(config)

        target_w = max(int(config.frame_target_width), 0)
        target_h = max(int(config.frame_target_height), 0)

        if target_w > 0 and target_h > 0:
            frame_proc = cv2.resize(frame, (target_w, target_h))
            clear_base = cv2.resize(clear_img, (target_w, target_h))
        else:
            frame_proc = frame
            clear_base = clear_img
            if not args.resize_clear and clear_img.shape[:2] != frame.shape[:2]:
                raise ValueError("Clear frame and target frame must share the same resolution or use --resize-clear.")

        if args.resize_clear:
            clear_for_frame = clear_base
        else:
            clear_for_frame = clear_base

        estimator.initialize(clear_for_frame)
        estimate = estimator.estimate(frame_proc, vehicle_boxes=[])

        compare_row, compare_dist, _, _, _ = compute_vis_compare_debug(estimator, frame_proc, estimate)
        trans_row, trans_dist, trans_mask = compute_vis_trans_debug(estimator, frame_proc, estimate)

        annotated = frame_proc.copy()
        draw_horizontal_line(
            annotated,
            compare_row,
            (0, 255, 0),
            f"vis_compare ~ {compare_dist:.1f} m",
        )

        if trans_row is not None and trans_dist is not None:
            draw_horizontal_line(
                annotated,
                trans_row,
                (0, 128, 255),
                f"vis_trans ~ {trans_dist:.1f} m",
            )

        cv2.putText(
            annotated,
            f"lambda={estimate.lambda_fused:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            lineType=cv2.LINE_AA,
        )

        out_path = output_dir / (Path(frame_path).stem + "_annotated.png")
        cv2.imwrite(str(out_path), annotated)

        if trans_mask is not None:
            mask_viz = (trans_mask * 255).astype(np.uint8)
            cv2.imwrite(str(output_dir / (Path(frame_path).stem + "_trans_mask.png")), mask_viz)

        trans_str = f"{trans_dist:.1f}" if trans_dist is not None else "nan"
        print(
            f"{frame_path}: vis_compare ~{compare_dist:.1f} m (row={compare_row}); "
            f"vis_trans ~{trans_str} m (row={trans_row})"
        )


if __name__ == "__main__":
    main()
