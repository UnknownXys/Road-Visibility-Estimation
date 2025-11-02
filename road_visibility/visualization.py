from __future__ import annotations

from pathlib import Path
from typing import Sequence, Tuple

import cv2
import numpy as np

from .types import LaneSegment, VisibilityEstimate


def draw_lane_segments(
    image: np.ndarray,
    segments: Sequence[LaneSegment],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    annotated = image.copy()
    for seg in segments:
        contour = seg.contour.astype(np.int32)
        if contour.ndim == 2:
            contour = contour[:, None, :]
        cv2.drawContours(annotated, [contour], -1, color, thickness)
        box = seg.bounding_box
        cv2.rectangle(
            annotated,
            (box.x, box.y),
            (box.x + box.width, box.y + box.height),
            color,
            1,
        )
        label = f"{seg.length_px:.1f}px"
        cv2.putText(
            annotated,
            label,
            (box.x, max(0, box.y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
            cv2.LINE_AA,
        )
    return annotated


def draw_vanishing_point(
    image: np.ndarray,
    col: float,
    row: float,
    color: Tuple[int, int, int] = (0, 0, 255),
    radius: int = 6,
    thickness: int = 2,
) -> np.ndarray:
    annotated = image.copy()
    center = (int(round(col)), int(round(row)))
    cv2.drawMarker(
        annotated,
        center,
        color,
        markerType=cv2.MARKER_CROSS,
        markerSize=radius * 2,
        thickness=thickness,
    )
    cv2.circle(annotated, center, radius, color, 1, cv2.LINE_AA)
    return annotated


def annotate_visibility(
    frame: np.ndarray,
    estimate: VisibilityEstimate,
    lane_color: Tuple[int, int, int] = (0, 255, 0),
    vanish_color: Tuple[int, int, int] = (0, 0, 255),
) -> np.ndarray:
    annotated = draw_lane_segments(frame, estimate.lane_segments, color=lane_color)
    annotated = draw_vanishing_point(
        annotated,
        estimate.vanish_point_col,
        estimate.vanish_point_row,
        color=vanish_color,
    )

    detect_text = (
        f"{estimate.visibility_detect:.1f}m"
        if estimate.visibility_detect is not None
        else "n/a"
    )
    fused_text = (
        f"{estimate.visibility_fused:.1f}m"
        if estimate.visibility_fused is not None
        else "n/a"
    )
    text = (
        f"vp=({estimate.vanish_point_col:.1f},{estimate.vanish_point_row:.1f}) "
        f"lambda={estimate.lambda_fused:.0f} "
        f"vis_edge={estimate.visibility_compare:.1f}m "
        f"vis_vehicle={detect_text} "
        f"vis_fused={fused_text}"
    )
    cv2.putText(
        annotated,
        text,
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return annotated
