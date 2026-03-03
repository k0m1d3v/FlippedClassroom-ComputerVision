import cv2
import numpy as np
from detectors.base import BaseDetector, Detection

class Detector(BaseDetector):
    name = "lanes_hough"

    def __init__(self):
        # parametri regolabili
        self.canny1 = 80
        self.canny2 = 160

    def detect(self, frame, frame_idx: int):
        h, w = frame.shape[:2]
        dets = []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, self.canny1, self.canny2)

        # ROI triangolare / trapezio nella parte bassa (strada)
        mask = np.zeros_like(edges)
        poly = np.array([[
            (int(w*0.10), h),
            (int(w*0.45), int(h*0.60)),
            (int(w*0.55), int(h*0.60)),
            (int(w*0.90), h),
        ]], dtype=np.int32)
        cv2.fillPoly(mask, poly, 255)
        roi = cv2.bitwise_and(edges, mask)

        lines = cv2.HoughLinesP(
            roi,
            rho=1,
            theta=np.pi/180,
            threshold=60,
            minLineLength=60,
            maxLineGap=80
        )

        if lines is None:
            return dets

        # raggruppo grezzo per pendenza (sinistra/destra)
        left = []
        right = []
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            if x2 == x1:
                continue
            slope = (y2 - y1) / (x2 - x1)
            if abs(slope) < 0.3:
                continue
            if slope < 0:
                left.append((x1, y1, x2, y2))
            else:
                right.append((x1, y1, x2, y2))

        # restituisco due “aree” come detection (solo per overlay)
        if left:
            dets.append(Detection(int(w*0.05), int(h*0.55), int(w*0.48), h, "lane_left", 0.7).clamp(w, h))
        if right:
            dets.append(Detection(int(w*0.52), int(h*0.55), int(w*0.95), h, "lane_right", 0.7).clamp(w, h))

        return dets