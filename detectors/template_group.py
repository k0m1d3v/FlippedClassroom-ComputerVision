import cv2
import numpy as np
from detectors.base import BaseDetector, Detection

class Detector(BaseDetector):
    name = "group_TEMPLATE"

    def __init__(self):
        # inizializza modelli/parametri qui
        # es: self.roi = (x1, y1, x2, y2)
        pass

    def detect(self, frame, frame_idx: int):
        h, w = frame.shape[:2]
        dets = []

        # ESEMPIO: preprocessing
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # blur = cv2.GaussianBlur(gray, (5,5), 0)

        # TODO: rilevamento qui
        # dets.append(Detection(x1, y1, x2, y2, "label", conf).clamp(w, h))

        return dets