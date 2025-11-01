import cv2
import numpy as np
from road_visibility.config import PipelineConfig
from road_visibility.detectors.lane_markings import LaneDashDetector

config = PipelineConfig()
img = cv2.imread('clear_refs/G1523K315354_clear.jpg')
det = LaneDashDetector(config)
height, width = img.shape[:2]
roi_mask = det._build_roi_mask(height, width, 70.0)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
mask = det._build_candidate_mask(gray, roi_mask)
cv2.imwrite('debug_dash_mask.png', mask)
points = [((140, 280), (181, 234)), ((211, 196), (230, 175))]
for (x1, y1), (x2, y2) in points:
    x_min = min(x1, x2)
    x_max = max(x1, x2)
    y_min = min(y1, y2)
    y_max = max(y1, y2)
    roi = mask[y_min : y_max + 1, x_min : x_max + 1]
    density = float(cv2.countNonZero(roi)) / float(roi.size) if roi.size else 0.0
    print((x1, y1), (x2, y2), 'density', density, 'sum', cv2.countNonZero(roi), 'size', roi.size)
