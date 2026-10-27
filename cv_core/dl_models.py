"""Lớp B — YOLO (detect) + MobileNetV3 (classify ROI)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .fusion import DetectorVote

import os
import sys

# Khắc phục lỗi DLL trên Windows Python 3.8+ (c10.dll, fbgemm.dll)
if sys.platform == "win32":
    try:
        import site
        for sp in site.getsitepackages():
            tlib = os.path.join(sp, "torch", "lib")
            if os.path.exists(tlib):
                try:
                    os.add_dll_directory(tlib)
                except Exception:
                    pass
    except Exception:
        pass

try:
    import torch
    import torch.nn as nn
    from torchvision import models, transforms

    TORCH_OK = True
except OSError as exc:
    msg = str(exc).encode("ascii", errors="replace").decode("ascii")
    print(f"[Torch] Loi DLL (thuong gap tren Windows): {msg}")
    print("[Torch] App van chay bang EAR/MAR. De train YOLO: cai VC++ Redistributable + pip install torch.")
    torch = None
    TORCH_OK = False
except Exception as exc:
    msg = str(exc).encode("ascii", errors="replace").decode("ascii")
    print(f"[Torch] Khong import duoc: {msg}")
    torch = None
    TORCH_OK = False

EYE_WEIGHTS = Path("runs/detecteye/train/weights/best.pt")
YAWN_WEIGHTS = Path("runs/detectyawn/train/weights/best.pt")
MOBILENET_EYE = Path("runs/classify_eye/mobilenet_eye.pt")
MOBILENET_MOUTH = Path("runs/classify_mouth/mobilenet_mouth.pt")
EFFICIENTNET_EYE = Path("runs/classify_eye/efficientnet_eye.pt")
EFFICIENTNET_MOUTH = Path("runs/classify_mouth/efficientnet_mouth.pt")

DL_CONF = 0.60


def build_mobilenet(num_classes: int = 2):
    if not TORCH_OK:
        raise RuntimeError("PyTorch chua chay duoc tren may nay.")
    net = models.mobilenet_v3_small(weights=None)
    in_f = net.classifier[-1].in_features
    net.classifier[-1] = nn.Linear(in_f, num_classes)
    return net


def build_efficientnet(num_classes: int = 2):
    if not TORCH_OK:
        raise RuntimeError("PyTorch chua chay duoc tren may nay.")
    net = models.efficientnet_b0(weights=None)
    in_f = net.classifier[-1].in_features
    net.classifier[-1] = nn.Linear(in_f, num_classes)
    return net


class _RoiClassifier:
    def __init__(self, weights: Path, builder, name: str, device: str):
        self.ok = False
        self.name = name
        self.device = device
        if not TORCH_OK:
            return
        self.tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        if not weights.exists():
            return
        try:
            self.model = builder(2)
            state = torch.load(weights, map_location=device)
            self.model.load_state_dict(state)
            self.model.to(device).eval()
            self.ok = True
            print(f"[{name}] Loaded {weights}")
        except Exception as exc:
            print(f"[{name}] Khong load duoc {weights}: {exc}")

    def predict(self, bgr_roi: np.ndarray) -> tuple[int, float]:
        if not TORCH_OK or not self.ok or bgr_roi is None or bgr_roi.size == 0:
            return -1, 0.0
        with torch.inference_mode():
            rgb = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2RGB)
            x = self.tf(rgb).unsqueeze(0).to(self.device)
            logits = self.model(x)
            prob = torch.softmax(logits, dim=1)[0]
            idx = int(torch.argmax(prob).item())
            return idx, float(prob[idx].item())


class DeepDetectors:
    """YOLO: class 0 = open/yawn?  Eye yaml: 0 open, 1 close. Yawn yaml: 0 yawn, 1 no yawn."""

    def __init__(self):
        self.eye_yolo = None
        self.yawn_yolo = None
        self.device = "cpu"
        if not TORCH_OK:
            print("[YOLO] Bo qua — PyTorch loi DLL. Dung EAR/MAR.")
            self.eye_cls = _RoiClassifier(MOBILENET_EYE, build_mobilenet, "MobileNet-eye", self.device)
            self.mouth_cls = _RoiClassifier(MOBILENET_MOUTH, build_mobilenet, "MobileNet-mouth", self.device)
            return
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            from ultralytics import YOLO

            if EYE_WEIGHTS.exists():
                self.eye_yolo = YOLO(str(EYE_WEIGHTS))
            else:
                print(f"[YOLO] Thieu {EYE_WEIGHTS} — dung EAR lam chinh.")
            if YAWN_WEIGHTS.exists():
                self.yawn_yolo = YOLO(str(YAWN_WEIGHTS))
            else:
                print(f"[YOLO] Thieu {YAWN_WEIGHTS} — dung MAR lam chinh.")
        except Exception as exc:
            print(f"[YOLO] Khong khoi tao duoc: {exc}")

        self.eye_cls = _RoiClassifier(MOBILENET_EYE, build_mobilenet, "MobileNet-eye", self.device)
        self.mouth_cls = _RoiClassifier(MOBILENET_MOUTH, build_mobilenet, "MobileNet-mouth", self.device)
        if not self.eye_cls.ok:
            self.eye_cls = _RoiClassifier(EFFICIENTNET_EYE, build_efficientnet, "EfficientNet-eye", self.device)
        if not self.mouth_cls.ok:
            self.mouth_cls = _RoiClassifier(EFFICIENTNET_MOUTH, build_efficientnet, "EfficientNet-mouth", self.device)

    def _yolo_best(self, model, roi: np.ndarray) -> tuple[int, float]:
        if model is None or roi is None or roi.size == 0:
            return -1, 0.0
        try:
            results = model.predict(roi, verbose=False, imgsz=320)[0]
            boxes = results.boxes
            if boxes is None or len(boxes) == 0:
                return -1, 0.0
            conf = boxes.conf.cpu().numpy()
            cls = boxes.cls.cpu().numpy()
            i = int(np.argmax(conf))
            return int(cls[i]), float(conf[i])
        except Exception:
            return -1, 0.0

    def vote_eyes(self, left_roi: np.ndarray | None, right_roi: np.ndarray | None) -> DetectorVote:
        votes_close = []
        votes_open = []

        for roi in (left_roi, right_roi):
            cid, conf = self._yolo_best(self.eye_yolo, roi)
            if cid == 1 and conf >= DL_CONF:
                votes_close.append(conf)
            elif cid == 0 and conf >= DL_CONF:
                votes_open.append(conf)
            mc, mconf = self.eye_cls.predict(roi)
            # MobileNet eye: 0 open, 1 close (cùng yaml)
            if mc == 1 and mconf >= DL_CONF:
                votes_close.append(mconf)
            elif mc == 0 and mconf >= DL_CONF:
                votes_open.append(mconf)

        if not votes_close and not votes_open:
            return DetectorVote(eye_closed=None, eye_conf=0.0)
        if len(votes_close) > len(votes_open):
            return DetectorVote(eye_closed=True, eye_conf=float(np.mean(votes_close)))
        if len(votes_open) > len(votes_close):
            return DetectorVote(eye_closed=False, eye_conf=float(np.mean(votes_open)))
        # Hòa: lấy nhánh conf cao hơn
        c_m, o_m = (float(np.mean(votes_close)) if votes_close else 0.0), (
            float(np.mean(votes_open)) if votes_open else 0.0
        )
        if c_m >= o_m:
            return DetectorVote(eye_closed=True, eye_conf=c_m)
        return DetectorVote(eye_closed=False, eye_conf=o_m)

    def vote_yawn(self, mouth_roi: np.ndarray | None) -> DetectorVote:
        cid, conf = self._yolo_best(self.yawn_yolo, mouth_roi)
        mc, mconf = self.mouth_cls.predict(mouth_roi)
        yawn_scores, rest_scores = [], []
        if cid == 0 and conf >= DL_CONF:
            yawn_scores.append(conf)
        elif cid == 1 and conf >= DL_CONF:
            rest_scores.append(conf)
        # MobileNet mouth: 0 yawn, 1 no yawn
        if mc == 0 and mconf >= DL_CONF:
            yawn_scores.append(mconf)
        elif mc == 1 and mconf >= DL_CONF:
            rest_scores.append(mconf)
        if not yawn_scores and not rest_scores:
            return DetectorVote(yawn=None, yawn_conf=0.0)
        if yawn_scores and (not rest_scores or np.mean(yawn_scores) >= np.mean(rest_scores)):
            return DetectorVote(yawn=True, yawn_conf=float(np.mean(yawn_scores)))
        return DetectorVote(yawn=False, yawn_conf=float(np.mean(rest_scores)))
