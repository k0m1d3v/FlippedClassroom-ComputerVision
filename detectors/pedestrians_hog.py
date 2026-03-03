import cv2
from detectors.base import BaseDetector, Detection

class Detector(BaseDetector):
    name = "pedestrians_hog"

    def __init__(self):
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def detect(self, frame, frame_idx: int):
        h, w = frame.shape[:2]
        dets = []

        scale = 0.75
        small = cv2.resize(frame, (0, 0), fx=scale, fy=scale)

        rects, weights = self.hog.detectMultiScale(
            small,
            winStride=(8, 8),
            padding=(8, 8),
            scale=1.05
        )

        for (x, y, rw, rh), conf in zip(rects, weights):
            x1 = int(x / scale)
            y1 = int(y / scale)
            x2 = int((x + rw) / scale)
            y2 = int((y + rh) / scale)
            dets.append(Detection(x1, y1, x2, y2, "pedestrian", float(conf)).clamp(w, h))

        return dets