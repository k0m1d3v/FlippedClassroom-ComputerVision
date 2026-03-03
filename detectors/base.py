from __future__ import annotations
from dataclasses import dataclass
from typing import List

@dataclass
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    label: str
    conf: float = 1.0

    def clamp(self, w: int, h: int) -> "Detection":
        self.x1 = max(0, min(self.x1, w - 1))
        self.y1 = max(0, min(self.y1, h - 1))
        self.x2 = max(0, min(self.x2, w - 1))
        self.y2 = max(0, min(self.y2, h - 1))
        if self.x2 < self.x1:
            self.x1, self.x2 = self.x2, self.x1
        if self.y2 < self.y1:
            self.y1, self.y2 = self.y2, self.y1
        return self

class BaseDetector:
    name: str = "base"

    def detect(self, frame, frame_idx: int) -> List[Detection]:
        return []