# Road Visibility Estimation (Python)

This project re-implements the visibility estimation logic from `src/visibility_detect.cpp` in Python. It follows the same high-level workflow:

1. Detect the road vanishing point.
2. Detect roadway dashed lane segments and measure their image-space length.
3. Calibrate the camera distance parameter `lambda` using the known physical lane segment length (6 m).
4. Optionally, estimate an alternative `lambda` from lightweight vehicle detections assuming a 4 m vehicle length.
5. Compare current foggy-scene edges with clear-day reference edges to compute `caculate_vis_compare`.
6. Use the furthest visible vehicle detection to compute `caculate_vis_detect`.

## Visibility estimation flowchart / 流程图

```
输入帧
  |
  v
LaneBoundaryDetector -> 道路 ROI + 灭点锁定
  |
  v
车辆历史 -> ROI 扩展
  |
  v
HybridLaneDetector -> 车道虚线段
  |
  v
λ 候选
  |- 虚线段 λ_lane
  |- 车辆 λ_vehicle
        |
        v
      λ 锁定与融合同步更新 last_lambda_candidate
        |
        v
    +-----------------------+----------------------------+
    | visibility_detect     | visibility_compare         |
    | 车辆上沿 -> 距离估计    | Canny 边缘 -> 背景对比       |
    | 最小值队列滤波         | 滑窗保持率 -> 最小值队列      |
    +-----------------------+----------------------------+
                        |
                        v
      compute_transmittance_metrics (可选) -> visibility_trans
                        |
                        v
            VisibilityEstimate 输出 (λ_fused, 各类距离, ROI)
```

## Project layout

```
python_visibility/
  README.md
  pyproject.toml
  road_visibility/
    __init__.py
    background.py
    camera_model.py
    config.py
    types.py
    utils.py
    visualization.py
    detectors/
      __init__.py
      base.py
      lane_markings.py
      lane_segmentation.py
      hybrid_lane.py
      vanishing_point.py
      vehicles.py
  scripts/
    estimate_visibility.py
    estimate_visibility_video.py
    test_lane_vp.py
  tools/
    check_segments.py
    debug_dash_density.py
    debug_dash_filters.py
    debug_lines.py
    debug_roi_visuals.py
    export_visibility_series.py
    inspect_lsd.py
    process_dash.py
    process_video_segment.py
    render_detection_overlay.py
    render_single_frame.py
```

### Utility scripts

All auxiliary debug and export helpers now live in `tools/`. Invoke them explicitly, for example:

```bash
python tools/render_detection_overlay.py --help
```

If you had local aliases or automation pointing at the previous `scripts/` paths, update them to reference `tools/` instead.

## Installation

```bash
python -m venv .venv
. .venv/Scripts/activate          # Windows (PowerShell)
pip install -e .
```

Add the optional Ultralytics YOLO dependency if you need automatic vehicle-based lambda estimation:

```bash
pip install -e .[yolo]
```

## Usage

The end-to-end estimator is exposed via the `RoadVisibilityEstimator` class:

```python
import cv2
from road_visibility.visibility import RoadVisibilityEstimator
from road_visibility.config import PipelineConfig

estimator = RoadVisibilityEstimator(PipelineConfig())

# Warm-up with a clear day frame (or a short sequence) to learn background edges.
clear_frame = cv2.imread("data/clear_frame.jpg")
estimator.initialize(clear_frame)

# Estimate visibility on a foggy frame.
fog_frame = cv2.imread("data/fog_frame.jpg")
result = estimator.estimate(fog_frame)

print(result.lambda_from_lane)     # camera distance parameter
print(result.visibility_compare)   # edge-based visibility
print(result.visibility_detect)    # vehicle-based visibility
```

Run the demo script for a self-contained example:

```bash
python scripts/estimate_visibility.py --clear data/clear.jpg --fog data/fog.jpg
```

### Optional lane segmentation model

The pipeline can fuse a lightweight lane-segmentation network with the geometric lane detector. Provide an ONNX model via the demo script:

```bash
python scripts/estimate_visibility.py \
    --clear data/clear.jpg \
    --fog data/fog.jpg \
    --lane-model checkpoints/lane.onnx \
    --lane-input-size 640x360
```

If the ONNX file or `onnxruntime` is unavailable, the estimator automatically falls back to the handcrafted detector, so you can iterate incrementally.

### Lane & vanishing-point visualisation

Use the dedicated tester to inspect lane-dash detections and the inferred vanishing point:

```bash
# Annotate a single image
python scripts/test_lane_vp.py --image path/to/frame.jpg --output-dir debug/lane_check

# Sample frames from a video (saves annotated JPEGs)
python scripts/test_lane_vp.py --video data/drive.mp4 --sample-stride 180 --max-samples 12 \
    --output-dir debug/video_check
```

## Video pipeline

When you have only a video, treat the first 20 % of frames as clear-scene input. The video interface automatically averages those frames to build the background model, with an option to inject an external clear image:

```bash
python scripts/estimate_visibility_video.py \
    --video data/drive.mp4 \
    --warmup-fraction 0.25 \
    --frame-stride 5 \
    --use-yolo \
    --lane-model checkpoints/lane.onnx
```

Add `--clear-image path/to/clear.jpg` if you want to override the automatically generated reference.

## Notes

- The vanishing point and lane-marking detection rely on image processing heuristics; tune `PipelineConfig` for your scene.
- If you have multiple clear-day frames, pass them to `initialize` sequentially to stabilise the background edge model.
- Vehicle detection uses an optional plugin system; register your own detector by subclassing `BaseVehicleDetector`.
- The output distances are in metres, assuming the provided physical dimensions (6 m lane segment, 4 m vehicle length).
