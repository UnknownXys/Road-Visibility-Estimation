"""
Road visibility estimation package mirroring the legacy C++ pipeline.
"""

from .visibility import RoadVisibilityEstimator, VisibilityEstimate
from .config import PipelineConfig
from .video import VideoVisibilityProcessor
from .types import FrameVisibility
from .visualization import annotate_visibility, draw_lane_segments, draw_vanishing_point

__all__ = [
    "RoadVisibilityEstimator",
    "VisibilityEstimate",
    "PipelineConfig",
    "VideoVisibilityProcessor",
    "FrameVisibility",
    "annotate_visibility",
    "draw_lane_segments",
    "draw_vanishing_point",
]
