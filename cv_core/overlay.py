"""Vẽ landmark, EAR/MAR, nhãn và cảnh báo lên frame."""

from __future__ import annotations

import cv2
import numpy as np

from .fusion import AlertLevel, FusionState
from .geometry import GeometryResult, LEFT_EYE, RIGHT_EYE, MOUTH


def draw_overlay(frame: np.ndarray, geo: GeometryResult | None, state: FusionState) -> np.ndarray:
    vis = frame.copy()
    if geo is not None:
        for idx in LEFT_EYE + RIGHT_EYE + MOUTH:
            x, y = int(geo.all_xy[idx, 0]), int(geo.all_xy[idx, 1])
            cv2.circle(vis, (x, y), 2, (0, 255, 255), -1)
        _poly(vis, geo.left_eye_pts, (0, 255, 0) if not state.eyes_closed else (0, 0, 255))
        _poly(vis, geo.right_eye_pts, (0, 255, 0) if not state.eyes_closed else (0, 0, 255))
        _poly(vis, geo.mouth_pts[:2], (255, 200, 0))

        ear_txt = f"EAR {geo.ear:.3f}  L{geo.ear_left:.3f} R{geo.ear_right:.3f}"
        mar_txt = f"MAR {geo.mar:.3f}  pitch {geo.pitch:.1f}"
        cv2.putText(vis, ear_txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 255, 50), 2)
        cv2.putText(vis, mar_txt, (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 200, 255), 2)
        eye_lbl = "CLOSE" if state.eyes_closed else "OPEN"
        yawn_lbl = "YAWN" if state.yawning else "NO-YAWN"
        cv2.putText(vis, f"Eyes: {eye_lbl}", (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255) if state.eyes_closed else (0, 255, 0), 2)
        cv2.putText(vis, f"Mouth: {yawn_lbl}", (10, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255) if state.yawning else (0, 255, 0), 2)
        cv2.putText(vis, f"PERCLOS {state.perclos_60*100:.1f}%", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        _perclos_bar(vis, state.perclos_60)
    else:
        cv2.putText(
            vis,
            "DANG TIM KIEM KHUON MAT...",
            (vis.shape[1] // 2 - 170, vis.shape[0] // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 215, 255),
            2,
        )

    if state.alert != AlertLevel.NONE:
        color = {
            AlertLevel.STRONG: (0, 0, 255),
            AlertLevel.FATIGUE: (0, 140, 255),
            AlertLevel.EARLY: (0, 215, 255),
        }[state.alert]
        cv2.rectangle(vis, (0, 0), (vis.shape[1] - 1, vis.shape[0] - 1), color, 8)
        cv2.putText(vis, state.alert_text[:48], (20, vis.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
    return vis


def _perclos_bar(img, perclos: float, x: int = 10, y: int = 132, w: int = 180, h: int = 12):
    p = max(0.0, min(1.0, float(perclos)))
    cv2.rectangle(img, (x, y), (x + w, y + h), (80, 80, 80), 1)
    fill = int(w * p)
    color = (0, 255, 0) if p < 0.15 else (0, 200, 255) if p < 0.30 else (0, 0, 255)
    if fill > 0:
        cv2.rectangle(img, (x, y), (x + fill, y + h), color, -1)
    cv2.putText(img, "PERCLOS", (x + w + 8, y + h), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)


def _poly(img, pts, color):
    p = np.round(pts).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [p], True, color, 1)
