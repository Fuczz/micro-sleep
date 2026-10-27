"""Lọc nhiễu thời gian (EMA + cửa sổ) cho EAR/MAR/pitch."""

from __future__ import annotations

from collections import deque


class TemporalSmoother:
    def __init__(self, alpha: float = 0.35, window: int = 7):
        self.alpha = alpha
        self.ear = None
        self.mar = None
        self.pitch = None
        self.yaw = None
        self.ear_win = deque(maxlen=window)
        self.mar_win = deque(maxlen=window)
        self.closed_votes = deque(maxlen=window)

    def update(
        self, ear: float, mar: float, pitch: float, yaw: float = 0.0
    ) -> tuple[float, float, float, float]:
        self.ear = ear if self.ear is None else self.alpha * ear + (1 - self.alpha) * self.ear
        self.mar = mar if self.mar is None else self.alpha * mar + (1 - self.alpha) * self.mar
        self.pitch = pitch if self.pitch is None else self.alpha * pitch + (1 - self.alpha) * self.pitch
        self.yaw = yaw if self.yaw is None else self.alpha * yaw + (1 - self.alpha) * self.yaw
        self.ear_win.append(self.ear)
        self.mar_win.append(self.mar)
        ear_m = sum(self.ear_win) / len(self.ear_win)
        mar_m = sum(self.mar_win) / len(self.mar_win)
        return ear_m, mar_m, self.pitch, self.yaw

    def majority_closed(self, closed: bool) -> bool:
        self.closed_votes.append(1 if closed else 0)
        return sum(self.closed_votes) > len(self.closed_votes) / 2
