from __future__ import annotations

import math
from typing import Optional

EPS_DELTA = 1e-3
EPS_DENOM = 1e-6


def lambda_from_segment(
    y_top: float,
    y_bottom: float,
    vanish_point_row: float,
    real_length_m: float,
) -> Optional[float]:
    """
    Estimate the camera distance parameter (lambda) from a vertical segment that
    spans the given pixel rows. Mirrors the computation found in the C++ code:

        lambda = L / (1/(y_top - vp) - 1/(y_bottom - vp))
    """

    if y_bottom <= y_top:
        return None
    top_delta = y_top - vanish_point_row
    bottom_delta = y_bottom - vanish_point_row
    if bottom_delta <= 0 or top_delta <= -5:
        return None

    safe_top = max(top_delta, EPS_DELTA)
    safe_bottom = max(bottom_delta, EPS_DELTA)
    denom = (1.0 / safe_top) - (1.0 / safe_bottom)
    if abs(denom) < EPS_DENOM:
        return None
    value = real_length_m / denom
    if not math.isfinite(value) or value < 0:
        return None
    return float(value)


def distance_from_row(lambda_value: float, vanish_point_row: float, row: float) -> float:
    delta = row - vanish_point_row
    if delta < 0:
        delta = 10.0
    return lambda_value / (0.1 + delta)
