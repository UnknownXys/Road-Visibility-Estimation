# Road Visibility Estimation


## Key Features

- **Hybrid λ calibration** - lane geometry (dashed line) with optional vehicle-length prior.  
- **Edge-based visibility (`vis_compare`)** - compares the current Canny edge map to a clear-scene background model using a sliding retention window.  
- **Transmittance-based visibility (`vis_trans`)** - estimates per-row transmittance via a dark-channel-style ratio, normalised by the clear reference.  
- **Vehicle distance upper bound (`vis_detect`)** - uses the nearest detected vehicle, when available, as a hard cap on visibility.  
- **Fused visibility output (`vis_fused`)** - weighted blend of the smoothed compare/trans signals with the optional vehicle upper bound.  
- Lightweight plotting & debugging tools.

## Repository Layout

```
python_visibility/
  road_visibility/
    config.py                 # PipelineConfig knobs (weights, smoothing, vehicle upper bound, etc.)
    visibility.py             # RoadVisibilityEstimator core logic
    visualization.py          # Drawing helpers for overlays
    types.py                  # Dataclasses (VisibilityEstimate now carries visibility_fused)
    detectors/                # Lane, boundary, vehicle detectors
  scripts/
    estimate_visibility.py
    estimate_visibility_video.py
  tools/
    debug_visibility_single.py
    render_single_frame.py
    export_visibility_series.py

```

## Installation

```bash
python -m venv .venv
# PowerShell:
. .venv/Scripts/Activate.ps1
pip install -e .
# Optional YOLO support:
pip install -e .[yolo]
```

## Quick Start

### Estimate visibility for a single frame

```bash
python scripts/estimate_visibility.py \
  --clear  data/clear_frame.jpg \
  --fog    data/foggy_frame.jpg
```

Example output:

```
[info] Initializing with clear frame...
  Vanish point: (168.5, 102.7)
[info] Estimating visibility on foggy frame...
  Visibility (edge compare): 206.2 m
  Visibility (transmittance): 98.9 m
  Visibility (fused): 174.0 m
```

### Process an entire video

```bash
python scripts/estimate_visibility_video.py \
  --video test.mp4 \
  --clear-image data/clear_frame.jpg \
  --frame-stride 10
```

The console progress line now includes the fused value:

```
[frame 00089] t=  2.97s vis_compare= 273.1m vis_trans= 112.1m fused= 174.0m
```

If `--clear-image` is omitted, the processor averages the first 60 seconds of footage to build the warm-up reference (override with `--warmup-fraction` if needed).

### Explore ROI and transmittance

```bash
python tools/debug_visibility_single.py \
  --clear clear_ref.png \
  --image t1.png \
  --save-trans-map debug/t1_trans_map.png
```

The script prints `vis_compare`, `vis_trans`, `vis_detect`, and the final `vis_fused`, and writes an overlay with compare/trans rows.

### Export frame-wise CSV

```bash
python tools/export_visibility_series.py \
  --video test.mp4 \
  --clear-image data/clear_frame.jpg \
  --output-dir results
```

CSV columns now include `visibility_fused_m` in addition to the legacy compare/detect metrics.

## Fusion Configuration Highlights

Tune `PipelineConfig` to match your scene:

| Field | Description |
| --- | --- |
| `visibility_compare_weight`, `visibility_trans_weight` | Blend weights between edge- and trans-based estimates. |
| `visibility_fusion_alpha_compare`, `visibility_fusion_alpha_trans`, `visibility_fusion_alpha_final` | EMA smoothing factors (set to 1.0 for raw values). |
| `visibility_min_distance` | Floor applied to every visibility signal before fusion. |
| `vehicle_upper_bound_window`, `vehicle_upper_bound_relax` | Controls how long the vehicle-derived upper bound is kept when detections disappear. |

All fusion parameters live in `road_visibility/config.py` and can be overridden when constructing `PipelineConfig`.

## Developer Notes

- `VisibilityEstimate.visibility_detect` is now `Optional[float]`; scripts and tools were updated to accept the `None` case.  
- The fused output is available as `VisibilityEstimate.visibility_fused`.  
- Vehicle-based visibility is treated strictly as an upper bound—no default fallback when detections are missing.  
