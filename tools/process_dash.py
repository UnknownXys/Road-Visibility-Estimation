import cv2
import numpy as np
import math

img = cv2.imread('clear_refs/G1523K315354_clear.jpg')
if img is None:
    raise RuntimeError('failed to load image')
height, width = img.shape[:2]

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (9, 9), 0)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
tophat = cv2.morphologyEx(blur, cv2.MORPH_TOPHAT, kernel)
adaptive = cv2.adaptiveThreshold(tophat, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, -5)
_, otsu = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
combined = cv2.bitwise_or(adaptive, otsu)
vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 15))
processed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, vert_kernel, iterations=1)
processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, vert_kernel, iterations=1)
processed = cv2.dilate(processed, vert_kernel, iterations=1)

roi_polygon = np.array([
    (int(width * 0.05), height - 1),
    (int(width * 0.92), height - 1),
    (int(width * 0.98), int(height * 0.22)),
    (int(width * 0.28), int(height * 0.22)),
], dtype=np.int32)
roi_mask = np.zeros((height, width), dtype=np.uint8)
cv2.fillConvexPoly(roi_mask, roi_polygon, 255)

masked = cv2.bitwise_and(processed, roi_mask)
contours, _ = cv2.findContours(masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

segments = []
for contour in contours:
    area = cv2.contourArea(contour)
    if area < 40 or area > 1400:
        continue

    x, y, w, h = cv2.boundingRect(contour)
    if h < 18 or h > 120:
        continue

    aspect = w / max(h, 1)
    if aspect > 0.65:
        continue

    mid_y = y + h * 0.5
    if mid_y < height * 0.3:
        continue

    length_norm = h / float(height)
    if length_norm < 0.05 or length_norm > 0.45:
        continue

    mid_x = x + w * 0.5
    col_norm = mid_x / float(width)
    if col_norm < 0.25 or col_norm > 0.9:
        continue

    top = (int(round(mid_x)), int(round(y)))
    bottom = (int(round(mid_x)), int(round(y + h)))
    segments.append((top, bottom, h))

segments.sort(key=lambda seg: seg[0][1])
print('segments', len(segments))
for seg in segments:
    print(seg)
