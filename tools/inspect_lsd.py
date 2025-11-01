import cv2
from road_visibility.config import PipelineConfig
from road_visibility.detectors.lane_markings import LaneDashDetector

config = PipelineConfig()
img = cv2.imread('clear_refs/G1523K315354_clear.jpg')
vanish_row = 70.0
height, width = img.shape[:2]

roi_mask = LaneDashDetector(config)._build_roi_mask(height, width, vanish_row)
masked_gray = cv2.bitwise_and(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), mask=roi_mask)
lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
lines = lsd.detect(masked_gray)[0]
print('total lines', 0 if lines is None else len(lines))
if lines is not None:
    for line in lines[:20]:
        print(line)
