"""Lớp A — đặc trưng hình học: EAR, MAR, head pose, ROI có padding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import cv2
import numpy as np

# MediaPipe Face Mesh (468 điểm) — 6 điểm EAR chuẩn
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
# Miệng: ngang (78-308) + 3 cặp dọc
MOUTH = [78, 308, 13, 14, 82, 312, 87, 317]
# solvePnP — 6 điểm cứng của khuôn mặt
POSE_IDX = [1, 152, 33, 263, 61, 291]
POSE_MODEL_3D = np.array(
    [
        (0.0, 0.0, 0.0),
        (0.0, -330.0, -65.0),
        (-225.0, 170.0, -135.0),
        (225.0, 170.0, -135.0),
        (-150.0, -150.0, -125.0),
        (150.0, -150.0, -125.0),
    ],
    dtype=np.float64,
)


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def eye_aspect_ratio(pts6: np.ndarray) -> float:
    """EAR = (||p2-p6|| + ||p3-p5||) / (2 ||p1-p4||)."""
    p1, p2, p3, p4, p5, p6 = pts6
    den = 2.0 * _dist(p1, p4)
    if den < 1e-6:
        return 0.0
    return (_dist(p2, p6) + _dist(p3, p5)) / den


def mouth_aspect_ratio(pts: np.ndarray) -> float:
    """MAR trên 8 điểm miệng MediaPipe."""
    left, right, t1, b1, t2, b2, t3, b3 = pts
    den = 2.0 * _dist(left, right)
    if den < 1e-6:
        return 0.0
    return (_dist(t1, b1) + _dist(t2, b2) + _dist(t3, b3)) / den


def rotation_to_euler(rvec: np.ndarray) -> tuple[float, float, float]:
    rot, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(rot[0, 0] ** 2 + rot[1, 0] ** 2)
    if sy < 1e-6:
        pitch = np.degrees(np.arctan2(-rot[1, 2], rot[1, 1]))
        yaw = np.degrees(np.arctan2(-rot[2, 0], sy))
        roll = 0.0
    else:
        pitch = np.degrees(np.arctan2(rot[2, 1], rot[2, 2]))
        yaw = np.degrees(np.arctan2(-rot[2, 0], sy))
        roll = np.degrees(np.arctan2(rot[1, 0], rot[0, 0]))
    return float(pitch), float(yaw), float(roll)


@dataclass
class GeometryResult:
    ear_left: float
    ear_right: float
    ear: float
    mar: float
    pitch: float
    yaw: float
    roll: float
    left_eye_pts: np.ndarray
    right_eye_pts: np.ndarray
    mouth_pts: np.ndarray
    all_xy: np.ndarray


class FaceGeometry:
    def landmarks_xy(self, landmarks, width: int, height: int) -> np.ndarray:
        xy = np.zeros((468, 2), dtype=np.float32)
        for i, lm in enumerate(landmarks.landmark[:468]):
            xy[i] = (lm.x * width, lm.y * height)
        return xy

    def compute(self, landmarks, width: int, height: int) -> Optional[GeometryResult]:
        xy = self.landmarks_xy(landmarks, width, height)
        left = xy[LEFT_EYE]
        right = xy[RIGHT_EYE]
        mouth = xy[MOUTH]
        ear_l = eye_aspect_ratio(left)
        ear_r = eye_aspect_ratio(right)
        mar = mouth_aspect_ratio(mouth)
        pitch, yaw, roll = self._head_pose(xy, width, height)
        return GeometryResult(
            ear_left=ear_l,
            ear_right=ear_r,
            ear=(ear_l + ear_r) / 2.0,
            mar=mar,
            pitch=pitch,
            yaw=yaw,
            roll=roll,
            left_eye_pts=left,
            right_eye_pts=right,
            mouth_pts=mouth,
            all_xy=xy,
        )

    def _head_pose(self, xy: np.ndarray, width: int, height: int) -> tuple[float, float, float]:
        image_pts = xy[POSE_IDX].astype(np.float64)
        cam = np.array(
            [[width, 0, width / 2.0], [0, width, height / 2.0], [0, 0, 1]],
            dtype=np.float64,
        )
        dist = np.zeros((4, 1))
        ok, rvec, _tvec = cv2.solvePnP(
            POSE_MODEL_3D, image_pts, cam, dist, flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not ok:
            return 0.0, 0.0, 0.0
        return rotation_to_euler(rvec)


def crop_with_padding(
    frame: np.ndarray,
    points: Sequence[np.ndarray],
    pad_ratio: float = 0.15,
    min_size: int = 12,
) -> Optional[np.ndarray]:
    """Cắt ROI theo min/max + padding 10–20%. Trả None nếu crop rỗng."""
    if frame is None or frame.size == 0 or len(points) == 0:
        return None
    pts = np.asarray(points, dtype=np.float32)
    x1, y1 = float(pts[:, 0].min()), float(pts[:, 1].min())
    x2, y2 = float(pts[:, 0].max()), float(pts[:, 1].max())
    w, h = x2 - x1, y2 - y1
    if w < 2 or h < 2:
        return None
    pad_x = max(int(w * pad_ratio), 2)
    pad_y = max(int(h * pad_ratio), 2)
    ih, iw = frame.shape[:2]
    xa = max(int(x1) - pad_x, 0)
    ya = max(int(y1) - pad_y, 0)
    xb = min(int(x2) + pad_x, iw)
    yb = min(int(y2) + pad_y, ih)
    if xb - xa < min_size or yb - ya < min_size:
        return None
    roi = frame[ya:yb, xa:xb]
    if roi.size == 0:
        return None
    return roi
