import cv2
import numpy as np
from road_visibility.config import PipelineConfig
from road_visibility.detectors.lane_markings import LaneDashDetector

config = PipelineConfig()
img = cv2.imread('clear_refs/G1523K315354_clear.jpg')
det = LaneDashDetector(config)
height, width = img.shape[:2]
vanish_row = 70.0
roi_mask = det._build_roi_mask(height, width, vanish_row)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
candidate_mask = det._build_candidate_mask(gray, roi_mask)
lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
lsd_lines = lsd.detect(cv2.bitwise_and(gray, gray, mask=roi_mask))[0]
hough_lines = cv2.HoughLinesP(candidate_mask, 1, np.pi/180, config.lane_hough_threshold, minLineLength=config.lane_hough_min_length, maxLineGap=config.lane_hough_max_gap)
lines = []
if lsd_lines is not None:
    lines.extend(lsd_lines.reshape(-1,4).tolist())
if hough_lines is not None:
    lines.extend(hough_lines.reshape(-1,4).tolist())
print('total lines', len(lines))
for line in lines:
    x1,y1,x2,y2 = line
    length = ( (x2-x1)**2 + (y2-y1)**2 ) ** 0.5
    dy = abs(y2-y1)
    angle = 90.0 if dy<1e-6 else abs(np.degrees(np.arctan((x2-x1)/dy)))
    if length > 40 and angle < 40:
        print('candidate', line, 'length', length, 'angle', angle)
