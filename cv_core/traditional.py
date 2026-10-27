"""Lớp C — SVM / Random Forest trên vector [EAR, MAR, pitch, blink_rate, PERCLOS]."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

SVM_PATH = Path("runs/traditional/drowsiness_svm.joblib")
RF_PATH = Path("runs/traditional/drowsiness_rf.joblib")
FEATURE_NAMES = ["ear", "mar", "pitch", "blink_rate", "perclos"]


class TraditionalDrowsinessModel:
    def __init__(self):
        self.model = None
        self.kind = None
        for path, kind in ((SVM_PATH, "SVM"), (RF_PATH, "RandomForest")):
            if path.exists():
                try:
                    self.model = joblib.load(path)
                    self.kind = kind
                    print(f"[ML] Loaded {kind} from {path}")
                    break
                except Exception as exc:
                    print(f"[ML] Loi load {path}: {exc}")

    def predict(self, ear: float, mar: float, pitch: float, blink_rate: float, perclos: float):
        if self.model is None:
            return None, 0.0
        x = np.array([[ear, mar, pitch, blink_rate, perclos]], dtype=np.float32)
        try:
            y = int(self.model.predict(x)[0])
            conf = 0.7
            if hasattr(self.model, "predict_proba"):
                conf = float(np.max(self.model.predict_proba(x)[0]))
            return bool(y == 1), conf
        except Exception:
            return None, 0.0
