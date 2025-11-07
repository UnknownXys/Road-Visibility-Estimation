from dataclasses import dataclass
from typing import Tuple


@dataclass
class PipelineConfig:
    """
    Configuration knobs for the Python visibility pipeline.
    Values are selected to mirror the heuristics in the C++ implementation.
    """

    lane_dash_length_m: float = 6.0
    vehicle_length_m: float = 4.0
    canny_thresholds: Tuple[int, int] = (40, 150)
    morph_kernel_size: int = 3
    frame_reference_width: int = 352
    frame_reference_height: int = 288
    frame_target_width: int = 352
    frame_target_height: int = 288
    lane_min_brightness: int = 175
    lane_roi_margin_px: int = 30
    lambda_trim_fraction: float = 0.15
    lambda_vehicle_update_alpha: float = 0.2
    lambda_vehicle_max_deviation: float = 0.3
    vanish_lock_alpha: float = 0.1
    vanish_tolerance_px: float = 30.0
    vanish_relock_frames: int = 60
    lane_min_area_px: int = 120
    lane_max_area_px: int = 8000
    lane_column_min_ratio: float = 0.15
    lane_column_max_ratio: float = 0.85
    lane_tophat_kernel: int = 9
    lane_vertical_kernel: int = 15
    lane_segment_min_len_px: float = 12.0
    lane_segment_max_len_px: float = 300.0
    lane_segment_max_single_len_px: float = 80.0
    lane_vertical_angle_min_deg: float = 20.0
    lane_vertical_angle_deg: float = 45.0
    lane_brightness_min_mean: float = 70.0
    lane_sample_width: int = 5
    lane_tophat_weight: float = 0.7
    lane_adaptive_threshold_block: int = 17
    lane_adaptive_threshold_c: int = -5
    lane_segment_min_density: float = 0.12
    lane_segment_length_norm_min: float = 0.04
    lane_segment_length_norm_max: float = 0.55
    lane_vanish_alignment_deg: float = 45.0
    lane_segment_min_continuous_ratio: float = 0.25
    lane_segment_max_gap_ratio: float = 0.15
    lane_contrast_threshold: float = 18.0
    lane_vanish_min_relative_row: float = 0.35
    lane_segment_dedup_epsilon_px: float = 25.0
    lane_clahe_clip: float = 2.0
    lane_clahe_grid: int = 8
    lane_hough_threshold: int = 30
    lane_hough_min_length: int = 14
    lane_hough_max_gap: int = 6
    boundary_min_line_length: int = 150
    boundary_angle_epsilon: float = 5.0
    boundary_min_separation_px: float = 50.0
    boundary_left_min_vertical_angle: float = 10.0
    boundary_right_max_vertical_angle: float = 60.0
    boundary_outlier_scale: float = 2.5
    boundary_weight_bottom_bias: float = 20.0
    boundary_length_weight_gamma: float = 1.5
    boundary_roi_expand_px: int = 70
    boundary_roi_tail_px: int = 130
    vehicle_roi_history_decay: float = 0.92
    vehicle_roi_column_threshold: float = 1.8
    vehicle_roi_expand_rows: int = 120
    vehicle_roi_expand_margin: int = 12
    vehicle_history_size: int = 2000
    vehicle_detection_interval: int = 5
    edge_window_rows: int = 12
    edge_required_stable: int = 5
    edge_min_reference_edges: int = 40
    edge_retention_threshold: float = 0.32
    edge_visibility_min_delta_rows: float = 25.0
    vis_compare_roi_expand_ratio: float = 0.08
    background_alpha: float = 1.0 / 30.0
    queue_size: int = 60
    vanish_point_offset_limit: int = 150
    default_lambda: float = 8000.0
    visibility_compare_max_distance: float = 1200.0
    roi_bottom_ratio: float = 0.98
    roi_top_ratio: float = 0.35
    roi_expand: int = 15
    transmittance_hist_bins: int = 32
    clear_scene_transmittance_threshold: float = 0.65
    clear_scene_min_gap_seconds: float = 5.0
    clear_scene_save_dir: str = "./clear_scenes"
    transmittance_gaussian_sigma: float = 1.5
    transmittance_hist_smoothing: int = 3
    transmittance_dcp_radius: int = 5
    transmittance_dcp_omega: float = 0.95
    transmittance_gamma: float = 1.0
    transmittance_contrast_low: float = 0.05
    transmittance_contrast_high: float = 0.995
    transmittance_contrast_blend: float = 0.3
    transmittance_atmosphere_top_percent: float = 0.001
    visibility_compare_weight: float = 0.7
    visibility_trans_weight: float = 0.3
    visibility_fusion_alpha_compare: float = 1.0
    visibility_fusion_alpha_trans: float = 1.0
    visibility_fusion_alpha_final: float = 1.0
    visibility_min_distance: float = 30.0
    vehicle_upper_bound_window: int = 60
    vehicle_upper_bound_relax: float = 1.02
    transmittance_row_base_fraction: float = 0.12
    transmittance_row_threshold_blend: float = 0.9
    transmittance_row_threshold_scale: float = 0.6
    transmittance_visibility_percentile: float = 0.95
    transmittance_visibility_min_pixels: int = 500
    transmittance_video_window: int = 60

