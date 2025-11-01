import cv2
import numpy as np
from road_visibility.config import PipelineConfig
from road_visibility.detectors.lane_markings import LaneDashDetector

config = PipelineConfig()
img = cv2.imread('clear_refs/G1523K315354_clear.jpg')
height, width = img.shape[:2]
vanish = 70.0

roi_mask = LaneDashDetector(config)._build_roi_mask(height, width, vanish)
mask_gray = cv2.bitwise_and(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), mask=roi_mask)
lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
lines = lsd.detect(mask_gray)[0]
print('lines', 0 if lines is None else len(lines))

candidate_mask = LaneDashDetector(config)._build_candidate_mask(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), roi_mask)

if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = map(float, line[0])
        mid_x = (x1 + x2) * 0.5
        mid_y = (y1 + y2) * 0.5
        reasons = []
        if mid_y <= vanish + config.lane_roi_margin_px:
            reasons.append('warm')
        if not (config.lane_column_min_ratio * width <= mid_x <= config.lane_column_max_ratio * width):
            reasons.append('column')
        length = float(np.hypot(x2 - x1, y2 - y1))
        if length < config.lane_segment_min_len_px or length > config.lane_segment_max_len_px:
            reasons.append('length')
        dy = abs(y2 - y1)
        if dy < 1e-3:
            reasons.append('horizontal')
        else:
            angle = abs(np.degrees(np.arctan((x2 - x1) / dy)))
            if angle > config.lane_vertical_angle_deg:
                reasons.append(f'angle={angle:.1f}')
        min_x = int(np.floor(min(x1, x2)))
        min_y = int(np.floor(min(y1, y2)))
        max_x = int(np.ceil(max(x1, x2)))
        max_y = int(np.ceil(max(y1, y2)))
        w_box = max(1, max_x - min_x)
        h_box = max(1, max_y - min_y)
        length_norm = h_box / float(height)
        if not (config.lane_segment_length_norm_min <= length_norm <= config.lane_segment_length_norm_max):
            reasons.append(f'length_norm={length_norm:.2f}')
        roi_slice = candidate_mask[min_y:max_y, min_x:max_x]
        if roi_slice.size == 0:
            density = 0.0
        else:
            density = float(cv2.countNonZero(roi_slice)) / float(roi_slice.size)
        if density < config.lane_segment_min_density:
            reasons.append(f'density={density:.2f}')
        brightness = LaneDashDetector(config)._sample_line_intensity(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (x1,y1),(x2,y2))
        if brightness < config.lane_brightness_min_mean:
            reasons.append(f'brightness={brightness:.1f}')
        if not reasons:
            print('PASS', (x1, y1, x2, y2))
        else:
            if min_y>200 and max_y>200:
                print('FAIL', (x1,y1,x2,y2), reasons)
