import cv2
from road_visibility.config import PipelineConfig
from road_visibility.detectors.lane_markings import LaneDashDetector

img = cv2.imread('clear_refs/G1523K315354_clear.jpg')
config = PipelineConfig()
segments = LaneDashDetector(config).detect(img, vanish_point_row=70.0)
print('segments', len(segments))
for seg in segments:
    print(seg.contour[0].tolist(), seg.contour[-1].tolist(), seg.length_px)
