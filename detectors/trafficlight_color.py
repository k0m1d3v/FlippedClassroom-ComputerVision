import cv2
import numpy as np
from detectors.base import BaseDetector, Detection

class Detector(BaseDetector):
    name = "trafficlight_color"

    def __init__(self):
        # ROI in alto (regola se serve)
        self.roi_y1 = 0.00
        self.roi_y2 = 0.45
        self.roi_x1 = 0.20
        self.roi_x2 = 0.80

    def _largest_blob_bbox(self, mask):
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        c = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area < 80:
            return None
        x, y, w, h = cv2.boundingRect(c)
        return x, y, x + w, y + h, area

    def detect(self, frame, frame_idx: int):
        H, W = frame.shape[:2]
        x1 = int(W * self.roi_x1); x2 = int(W * self.roi_x2)
        y1 = int(H * self.roi_y1); y2 = int(H * self.roi_y2)
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return []

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # rosso (due range in HSV)
        red1 = cv2.inRange(hsv, (0, 70, 70), (10, 255, 255))
        red2 = cv2.inRange(hsv, (170, 70, 70), (180, 255, 255))
        red = cv2.bitwise_or(red1, red2)

        # verde
        green = cv2.inRange(hsv, (40, 70, 70), (90, 255, 255))

        # pulizia
        k = np.ones((5, 5), np.uint8)
        red = cv2.morphologyEx(red, cv2.MORPH_OPEN, k)
        green = cv2.morphologyEx(green, cv2.MORPH_OPEN, k)

        dets = []

        rb = self._largest_blob_bbox(red)
        gb = self._largest_blob_bbox(green)

        # scegli il blob più “convincente”
        if rb and (not gb or rb[4] >= gb[4]):
            bx1, by1, bx2, by2, area = rb
            dets.append(Detection(x1+bx1, y1+by1, x1+bx2, y1+by2, "traffic_light_red", 0.65).clamp(W, H))
        elif gb:
            bx1, by1, bx2, by2, area = gb
            dets.append(Detection(x1+bx1, y1+by1, x1+bx2, y1+by2, "traffic_light_green", 0.65).clamp(W, H))

        return dets