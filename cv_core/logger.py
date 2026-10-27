"""Ghi CSV phục vụ báo cáo: EAR, MAR, pitch, PERCLOS, cảnh báo, FPS."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path


class SessionLogger:
    HEADER = [
        "timestamp",
        "dt_s",
        "fps",
        "face_ok",
        "ear",
        "mar",
        "pitch",
        "yaw",
        "eyes_closed",
        "yawning",
        "blinks",
        "yawns",
        "microsleep_s",
        "yawn_duration_s",
        "perclos_60",
        "blink_rate",
        "nod_count",
        "alert",
        "alert_text",
        "svm_drowsy",
    ]

    def __init__(self, log_dir: str = "logs"):
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = Path(log_dir) / f"session_{stamp}.csv"
        self._fh = open(self.path, "w", newline="", encoding="utf-8")
        self._w = csv.writer(self._fh)
        self._w.writerow(self.HEADER)
        self._fh.flush()

    def write(self, row: dict) -> None:
        try:
            if hasattr(self, "_fh") and not self._fh.closed:
                self._w.writerow([row.get(k, "") for k in self.HEADER])
        except Exception:
            pass

    def close(self) -> None:
        try:
            if hasattr(self, "_fh") and not self._fh.closed:
                self._fh.flush()
                self._fh.close()
        except Exception:
            pass
