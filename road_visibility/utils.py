from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np


def ensure_directory(path: str | os.PathLike) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


def polygon_from_vanish_point(
    width: int,
    height: int,
    vanish_point_row: float,
    top_ratio: float,
    bottom_ratio: float,
    expand_px: int,
) -> List[Tuple[int, int]]:
    top_y = int(vanish_point_row)
    bottom_y = int(clamp(height * bottom_ratio, top_y + 1, height - 1))
    top_width = int(width * top_ratio)
    half_top = top_width // 2
    cx = width // 2
    left_top = clamp(cx - half_top, 0, width - 1)
    right_top = clamp(cx + half_top, 0, width - 1)
    left_bottom = clamp(cx - half_top - expand_px, 0, width - 1)
    right_bottom = clamp(cx + half_top + expand_px, 0, width - 1)

    return [
        (int(left_bottom), bottom_y),
        (int(right_bottom), bottom_y),
        (int(right_top), top_y),
        (int(left_top), top_y),
    ]


def sliding_window_mean(values: Sequence[float]) -> float:
    clean = [v for v in values if math.isfinite(v)]
    if not clean:
        return float("nan")
    return float(sum(clean) / len(clean))


def trim_mean(values: List[float], trim_fraction: float) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    n = len(values)
    k = int(n * trim_fraction)
    if k * 2 >= n:
        return float(sum(values) / max(n, 1))
    trimmed = values[k : n - k]
    return float(sum(trimmed) / len(trimmed))


def read_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def dilate_erode(image: np.ndarray, kernel_size: int) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    dilated = cv2.dilate(image, kernel)
    eroded = cv2.erode(dilated, kernel)
    return eroded


def count_edges(image: np.ndarray) -> int:
    return int(cv2.countNonZero(image))


def compute_intersections(lines: Iterable[Tuple[float, float]]) -> List[Tuple[float, float]]:
    intersections: List[Tuple[float, float]] = []
    lines = list(lines)
    for i in range(len(lines)):
        rho1, theta1 = lines[i]
        for j in range(i + 1, len(lines)):
            rho2, theta2 = lines[j]
            det = math.cos(theta1) * math.sin(theta2) - math.sin(theta1) * math.cos(theta2)
            if abs(det) < 1e-4:
                continue
            x = (math.sin(theta2) * rho1 - math.sin(theta1) * rho2) / det
            y = (-math.cos(theta2) * rho1 + math.cos(theta1) * rho2) / det
            if math.isfinite(x) and math.isfinite(y):
                intersections.append((x, y))
    return intersections


def median_point(points: Sequence[Tuple[float, float]]) -> Tuple[float, float]:
    xs = sorted(p[0] for p in points)
    ys = sorted(p[1] for p in points)
    mid = len(points) // 2
    if len(points) % 2 == 1:
        return xs[mid], ys[mid]
    return (xs[mid - 1] + xs[mid]) * 0.5, (ys[mid - 1] + ys[mid]) * 0.5


def to_int_point(point: Tuple[float, float]) -> Tuple[int, int]:
    return int(round(point[0])), int(round(point[1]))
