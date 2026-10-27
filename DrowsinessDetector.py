"""Phát hiện buồn ngủ realtime: Lớp A (hình học) + B (YOLO/MobileNet) + C (SVM/RF)
Tích hợp:
- Cân sáng thích ứng (Adaptive Gamma) + Khử nhiễu bảo toàn cạnh (Bilateral Filter).
- Cơ chế Fallback khi cabin thiếu sáng hoặc mất dấu tài xế.
- Hệ thống âm thanh cảnh báo đa tầng theo mức độ nguy hiểm.
- Giao diện điều chỉnh tham số trực tiếp (Dynamic UI) + Tự động cân chỉnh (Auto-Calibration).
"""

from __future__ import annotations

import os
import queue
import sys
import threading
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Khắc phục triệt để lỗi C++ Unicode (PyQt5 platform plugin & MediaPipe binarypb)
# khi đường dẫn chứa tiếng Việt có dấu:
try:
    __file__.encode("ascii")
except (UnicodeEncodeError, UnicodeDecodeError):
    import subprocess
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    subprocess.run(["subst", "Z:", "/d"], capture_output=True)
    res_subst = subprocess.run(["subst", "Z:", curr_dir], capture_output=True)
    if res_subst.returncode == 0:
        py_exe = os.path.join("Z:", "venv", "Scripts", "python.exe")
        if not os.path.exists(py_exe):
            py_exe = sys.executable
        target_script = os.path.join("Z:", os.path.basename(__file__))
        exit_code = subprocess.call([py_exe, target_script] + sys.argv[1:])
        sys.exit(exit_code)

# Khắc phục xung đột OpenMP DLL giữa PyTorch và OpenCV trên Windows (WinError 1114):
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
except Exception:
    torch = None

import PyQt5
qt_plugin_dir = os.path.join(os.path.dirname(PyQt5.__file__), "Qt5", "plugins", "platforms")
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = qt_plugin_dir

import cv2
import mediapipe as mp
import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from cv_core import (
    AlertFusion,
    AlertLevel,
    DeepDetectors,
    FaceGeometry,
    SessionLogger,
    TemporalSmoother,
    TraditionalDrowsinessModel,
    analyze_lighting,
    crop_with_padding,
    draw_overlay,
    enhance_lighting,
)
from cv_core.fusion import FusionState


def play_alert_sound(alert: AlertLevel, audio_enabled: bool = True):
    """Phát âm thanh cảnh báo đa tầng theo mức độ nguy hiểm."""
    if not audio_enabled or alert == AlertLevel.NONE:
        return
    try:
        import winsound
        if alert == AlertLevel.STRONG:
            # Nguy hiểm cao (Microsleep / Mất dấu kéo dài): 3 nhịp dồn dập tần số cao
            winsound.Beep(1800, 140)
            winsound.Beep(2200, 160)
            winsound.Beep(1800, 140)
        elif alert == AlertLevel.FATIGUE:
            # Mệt mỏi tích lũy (PERCLOS > 30%): 2 nhịp cảnh báo
            winsound.Beep(1300, 160)
            winsound.Beep(1300, 160)
        elif alert == AlertLevel.EARLY:
            # Cảnh báo sớm (ngáp, gật nhẹ, quay mặt): 1 nhịp nhẹ
            winsound.Beep(900, 140)
    except Exception:
        print("\a", end="", flush=True)


class DrowsinessDetector(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DMS — Giám Sát Tài Xế Real-time (Xử Lý Ảnh Nâng Cao)")
        self.setGeometry(40, 40, 1380, 640)
        self.setStyleSheet("""
            QMainWindow { background-color: #F4F6F9; }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #D0D7DE;
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 10px;
                background-color: #FFFFFF;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
                color: #1A73E8;
            }
            QPushButton {
                background-color: #1A73E8;
                color: white;
                font-weight: bold;
                border-radius: 4px;
                padding: 6px 12px;
            }
            QPushButton:hover { background-color: #1557B0; }
            QPushButton:disabled { background-color: #CCCCCC; }
        """)

        # Tùy chọn hệ thống
        self.audio_enabled = True
        self.use_adaptive_gamma = True
        self.use_bilateral = True

        # Trạng thái Auto-Calibration
        self._calibrating = False
        self._calib_start_time = 0.0
        self._calib_samples: list[float] = []

        # Core logic
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.geometry = FaceGeometry()
        self.smoother = TemporalSmoother()
        self.fusion = AlertFusion()
        self.deep = DeepDetectors()
        self.ml = TraditionalDrowsinessModel()
        self.logger = SessionLogger()

        self._lock = threading.Lock()
        self._display_bgr = None
        self._state = FusionState()
        self._ear = self._mar = self._pitch = self._yaw = 0.0
        self._fps = 0.0
        self._face_ok = False
        self._dt = 0.0
        self._prev_perclos = 0.0
        self._prev_blink_rate = 0.0
        self._mean_brightness = 0.0
        self._is_too_dark = False

        # Thiết lập UI layout 3 phần: [Video] - [Metrics & Alert] - [Control Panel]
        self._init_ui()

        # Camera: Dùng DirectShow (CAP_DSHOW) để khắc phục lỗi MSMF can't grab frame (-1072875772)
        self.cap = self._init_camera()
        time.sleep(0.3)

        self.frame_queue = queue.Queue(maxsize=2)
        self.stop_event = threading.Event()
        self.capture_thread = threading.Thread(target=self.capture_frames, daemon=True)
        self.process_thread = threading.Thread(target=self.process_frames, daemon=True)
        self.capture_thread.start()
        self.process_thread.start()

        self.ui_timer = QTimer(self)
        self.ui_timer.timeout.connect(self.refresh_ui)
        self.ui_timer.start(50)

    def _init_ui(self):
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        main_layout = QHBoxLayout(self.central_widget)
        main_layout.setSpacing(12)

        # 1. Cột Video (640x480)
        self.video_label = QLabel(self)
        self.video_label.setStyleSheet("border: 2px solid #2C3E50; background-color: #000000; border-radius: 4px;")
        self.video_label.setFixedSize(640, 480)
        main_layout.addWidget(self.video_label)

        # 2. Cột Thông số & Cảnh báo
        self.info_label = QLabel()
        self.info_label.setStyleSheet(
            "background-color: white; border: 1px solid #D0D7DE; border-radius: 6px; padding: 10px;"
        )
        self.info_label.setMinimumWidth(360)
        main_layout.addWidget(self.info_label)

        # 3. Cột Bảng điều khiển (Control Panel)
        ctrl_panel = QWidget()
        ctrl_panel.setFixedWidth(330)
        ctrl_layout = QVBoxLayout(ctrl_panel)
        ctrl_layout.setContentsMargins(0, 0, 0, 0)
        ctrl_layout.setSpacing(8)

        # Group 1: Tự động cân chỉnh
        calib_group = QGroupBox("Tự Động Cân Chỉnh (Calibration)")
        calib_inner = QVBoxLayout(calib_group)
        calib_desc = QLabel("Nhìn thẳng camera 3s để học kích thước mắt tự nhiên của bạn:")
        calib_desc.setWordWrap(True)
        calib_desc.setStyleSheet("color: #555; font-size: 11px;")
        calib_inner.addWidget(calib_desc)

        self.btn_calib = QPushButton("▶ Bắt đầu cân chỉnh (3s)")
        self.btn_calib.clicked.connect(self.start_calibration)
        calib_inner.addWidget(self.btn_calib)

        self.calib_bar = QProgressBar()
        self.calib_bar.setRange(0, 100)
        self.calib_bar.setValue(0)
        self.calib_bar.setTextVisible(False)
        self.calib_bar.setFixedHeight(8)
        calib_inner.addWidget(self.calib_bar)

        self.calib_status_lbl = QLabel("Trạng thái: Sẵn sàng")
        self.calib_status_lbl.setStyleSheet("color: #2E7D32; font-size: 11px; font-weight: bold;")
        calib_inner.addWidget(self.calib_status_lbl)
        ctrl_layout.addWidget(calib_group)

        # Group 2: Điều chỉnh thông số trực tiếp
        params_group = QGroupBox("Điều Chỉnh Tham Số Trực Tiếp")
        params_inner = QVBoxLayout(params_group)

        # Slider EAR Close
        ear_box = QHBoxLayout()
        ear_lbl = QLabel("EAR Nhắm:")
        self.ear_val_lbl = QLabel("0.21")
        self.ear_val_lbl.setStyleSheet("font-weight: bold; color: #D32F2F;")
        ear_box.addWidget(ear_lbl)
        ear_box.addStretch()
        ear_box.addWidget(self.ear_val_lbl)
        params_inner.addLayout(ear_box)

        self.slider_ear = QSlider(Qt.Horizontal)
        self.slider_ear.setRange(12, 32)
        self.slider_ear.setValue(21)
        self.slider_ear.valueChanged.connect(self._on_ear_slider_changed)
        params_inner.addWidget(self.slider_ear)

        # Slider MAR Yawn
        mar_box = QHBoxLayout()
        mar_lbl = QLabel("MAR Ngáp:")
        self.mar_val_lbl = QLabel("0.55")
        self.mar_val_lbl.setStyleSheet("font-weight: bold; color: #F57C00;")
        mar_box.addWidget(mar_lbl)
        mar_box.addStretch()
        mar_box.addWidget(self.mar_val_lbl)
        params_inner.addLayout(mar_box)

        self.slider_mar = QSlider(Qt.Horizontal)
        self.slider_mar.setRange(35, 80)
        self.slider_mar.setValue(55)
        self.slider_mar.valueChanged.connect(self._on_mar_slider_changed)
        params_inner.addWidget(self.slider_mar)

        # Slider Microsleep
        micro_box = QHBoxLayout()
        micro_lbl = QLabel("Ngưỡng Microsleep:")
        self.micro_val_lbl = QLabel("1.0s")
        self.micro_val_lbl.setStyleSheet("font-weight: bold; color: #C2185B;")
        micro_box.addWidget(micro_lbl)
        micro_box.addStretch()
        micro_box.addWidget(self.micro_val_lbl)
        params_inner.addLayout(micro_box)

        self.slider_micro = QSlider(Qt.Horizontal)
        self.slider_micro.setRange(5, 25)
        self.slider_micro.setValue(10)
        self.slider_micro.valueChanged.connect(self._on_micro_slider_changed)
        params_inner.addWidget(self.slider_micro)

        # Slider YOLO/DL Confidence
        conf_box = QHBoxLayout()
        conf_lbl = QLabel("Độ tin cậy DL:")
        self.conf_val_lbl = QLabel("0.60")
        self.conf_val_lbl.setStyleSheet("font-weight: bold; color: #1976D2;")
        conf_box.addWidget(conf_lbl)
        conf_box.addStretch()
        conf_box.addWidget(self.conf_val_lbl)
        params_inner.addLayout(conf_box)

        self.slider_conf = QSlider(Qt.Horizontal)
        self.slider_conf.setRange(30, 90)
        self.slider_conf.setValue(60)
        self.slider_conf.valueChanged.connect(self._on_conf_slider_changed)
        params_inner.addWidget(self.slider_conf)

        ctrl_layout.addWidget(params_group)

        # Group 3: Cài đặt hệ thống & Tùy chọn xử lý ảnh
        opt_group = QGroupBox("Cấu Hình Tính Năng")
        opt_inner = QVBoxLayout(opt_group)

        self.cb_audio = QCheckBox("Bật âm thanh cảnh báo đa tầng")
        self.cb_audio.setChecked(True)
        self.cb_audio.toggled.connect(lambda v: setattr(self, "audio_enabled", v))
        opt_inner.addWidget(self.cb_audio)

        self.cb_gamma = QCheckBox("Cân sáng Adaptive Gamma")
        self.cb_gamma.setChecked(True)
        self.cb_gamma.toggled.connect(lambda v: setattr(self, "use_adaptive_gamma", v))
        opt_inner.addWidget(self.cb_gamma)

        self.cb_bilateral = QCheckBox("Khử nhiễu bảo toàn cạnh (Bilateral)")
        self.cb_bilateral.setChecked(True)
        self.cb_bilateral.toggled.connect(lambda v: setattr(self, "use_bilateral", v))
        opt_inner.addWidget(self.cb_bilateral)

        self.btn_reset = QPushButton("Đặt lại mặc định")
        self.btn_reset.setStyleSheet("background-color: #607D8B;")
        self.btn_reset.clicked.connect(self.reset_defaults)
        opt_inner.addWidget(self.btn_reset)

        ctrl_layout.addWidget(opt_group)
        ctrl_layout.addStretch()
        main_layout.addWidget(ctrl_panel)

    def start_calibration(self):
        """Kích hoạt chế độ đo đạc và cân chỉnh tự động."""
        self._calibrating = True
        self._calib_start_time = time.time()
        self._calib_samples.clear()
        self.btn_calib.setEnabled(False)
        self.calib_bar.setValue(0)
        self.calib_status_lbl.setText("Đang ghi nhận... Nhìn thẳng camera!")
        self.calib_status_lbl.setStyleSheet("color: #E65100; font-size: 11px; font-weight: bold;")

    def _on_ear_slider_changed(self, val: int):
        ear_close = val / 100.0
        ear_open = ear_close + 0.04
        self.ear_val_lbl.setText(f"{ear_close:.2f}")
        self.fusion.set_thresholds(ear_close=ear_close, ear_open=ear_open)

    def _on_mar_slider_changed(self, val: int):
        mar_yawn = val / 100.0
        mar_rest = max(0.20, mar_yawn - 0.15)
        self.mar_val_lbl.setText(f"{mar_yawn:.2f}")
        self.fusion.set_thresholds(mar_yawn=mar_yawn, mar_rest=mar_rest)

    def _on_micro_slider_changed(self, val: int):
        micro_s = val / 10.0
        self.micro_val_lbl.setText(f"{micro_s:.1f}s")
        self.fusion.set_thresholds(microsleep_s=micro_s)

    def _on_conf_slider_changed(self, val: int):
        conf = val / 100.0
        self.conf_val_lbl.setText(f"{conf:.2f}")
        self.fusion.set_thresholds(dl_conf=conf)

    def reset_defaults(self):
        """Khôi phục các thiết lập chuẩn ban đầu."""
        self.slider_ear.setValue(21)
        self.slider_mar.setValue(55)
        self.slider_micro.setValue(10)
        self.slider_conf.setValue(60)
        self.cb_audio.setChecked(True)
        self.cb_gamma.setChecked(True)
        self.cb_bilateral.setChecked(True)
        self.fusion.set_thresholds(
            ear_close=0.21,
            ear_open=0.25,
            mar_yawn=0.55,
            mar_rest=0.40,
            microsleep_s=1.0,
            dl_conf=0.60,
        )
        self.calib_status_lbl.setText("Đã đặt lại về cấu hình mặc định")
        self.calib_status_lbl.setStyleSheet("color: #2E7D32; font-size: 11px; font-weight: bold;")

    def _init_camera(self):
        """Khởi tạo camera với backend DirectShow để khắc phục triệt để lỗi MSMF grabFrame trên Windows."""
        for idx in (0, 1, 2):
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                # Warm-up vài frames đầu để cảm biến webcam mở khẩu độ / đo sáng
                frame = None
                ret = False
                for _ in range(5):
                    ret, frame = cap.read()
                    time.sleep(0.04)
                if ret and frame is not None and frame.size > 0:
                    print(f"[Camera] Da ket noi Webcam index {idx} qua DirectShow (CAP_DSHOW).")
                    return cap
                cap.release()

        # Fallback thử backend mặc định
        for idx in (0, 1):
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                frame = None
                ret = False
                for _ in range(5):
                    ret, frame = cap.read()
                    time.sleep(0.04)
                if ret and frame is not None and frame.size > 0:
                    print(f"[Camera] Da ket noi Webcam index {idx} qua Default Backend.")
                    return cap
                cap.release()

        print("[Camera] Khong tim thay camera kha dung, mo mac dinh index 0.")
        return cv2.VideoCapture(0, cv2.CAP_DSHOW)

    def capture_frames(self):
        while not self.stop_event.is_set():
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            self.frame_queue.put(frame)

    def process_frames(self):
        t_prev = time.time()
        while not self.stop_event.is_set():
            try:
                frame = self.frame_queue.get(timeout=1)
            except queue.Empty:
                continue

            now = time.time()
            dt = max(now - t_prev, 1e-3)
            t_prev = now
            fps = 1.0 / dt

            # 1. Đo mức sáng môi trường
            light_info = analyze_lighting(frame)
            is_too_dark = bool(light_info["is_too_dark"])
            mean_brightness = float(light_info["mean_brightness"])

            # 2. Tiền xử lý ảnh nâng cao: Cân trắng + Adaptive Gamma + CLAHE + Bilateral Filter
            enhanced = enhance_lighting(
                frame,
                clip_limit=2.0,
                use_adaptive_gamma=self.use_adaptive_gamma,
                use_bilateral=self.use_bilateral,
            )

            # 3. Nhận diện mốc khuôn mặt
            rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
            mesh = self.face_mesh.process(rgb)
            ih, iw = enhanced.shape[:2]

            geo = None
            face_ok = False
            votes = []
            svm_flag = None

            if mesh.multi_face_landmarks:
                face_ok = True
                geo = self.geometry.compute(mesh.multi_face_landmarks[0], iw, ih)
                ear, mar, pitch, yaw = self.smoother.update(geo.ear, geo.mar, geo.pitch, geo.yaw)

                # Thu thập mẫu hiệu chỉnh nếu đang trong phiên Auto-Calibration
                if self._calibrating:
                    self._calib_samples.append(ear)

                left_roi = crop_with_padding(enhanced, geo.left_eye_pts)
                right_roi = crop_with_padding(enhanced, geo.right_eye_pts)
                mouth_roi = crop_with_padding(enhanced, geo.mouth_pts)

                eye_vote = self.deep.vote_eyes(left_roi, right_roi)
                yawn_vote = self.deep.vote_yawn(mouth_roi)
                votes = [eye_vote, yawn_vote]
                svm_flag, _ = self.ml.predict(
                    ear, mar, pitch, self._prev_blink_rate, self._prev_perclos
                )
            else:
                ear = mar = pitch = yaw = 0.0

            # 4. Hợp nhất cảnh báo (Hỗ trợ Fallback khi thiếu sáng / mất khuôn mặt)
            state = self.fusion.update(
                now=now,
                ear=ear,
                mar=mar,
                pitch=pitch,
                votes=votes,
                svm_drowsy=svm_flag,
                face_ok=face_ok,
                yaw=yaw,
                is_too_dark=is_too_dark,
            )
            self._prev_perclos = state.perclos_60
            self._prev_blink_rate = state.blink_rate

            # 5. Vẽ overlay trực quan
            vis = draw_overlay(enhanced, geo, state)

            # 6. Kích hoạt âm thanh đa tầng nếu có cảnh báo
            if self.fusion.should_beep(now):
                threading.Thread(
                    target=play_alert_sound,
                    args=(state.alert, self.audio_enabled),
                    daemon=True,
                ).start()

            # 7. Ghi nhật ký phân tích
            self.logger.write(
                {
                    "timestamp": f"{now:.3f}",
                    "dt_s": f"{dt:.4f}",
                    "fps": f"{fps:.1f}",
                    "face_ok": int(face_ok),
                    "mean_brightness": f"{mean_brightness:.1f}",
                    "is_too_dark": int(is_too_dark),
                    "ear": f"{ear:.4f}",
                    "mar": f"{mar:.4f}",
                    "pitch": f"{pitch:.2f}",
                    "yaw": f"{yaw:.2f}",
                    "eyes_closed": int(state.eyes_closed),
                    "yawning": int(state.yawning),
                    "blinks": state.blinks,
                    "yawns": state.yawns,
                    "microsleep_s": f"{state.microsleep_s:.3f}",
                    "yawn_duration_s": f"{state.yawn_duration_s:.3f}",
                    "perclos_60": f"{state.perclos_60:.4f}",
                    "blink_rate": f"{state.blink_rate:.1f}",
                    "nod_count": state.nod_count,
                    "alert": state.alert.value,
                    "alert_text": state.alert_text,
                    "svm_drowsy": "" if svm_flag is None else int(svm_flag),
                }
            )

            with self._lock:
                self._display_bgr = vis
                self._state = state
                self._ear, self._mar, self._pitch, self._yaw = ear, mar, pitch, yaw
                self._fps = fps
                self._face_ok = face_ok
                self._dt = dt
                self._mean_brightness = mean_brightness
                self._is_too_dark = is_too_dark

    def refresh_ui(self):
        # 1. Cập nhật tiến độ Auto-Calibration nếu đang chạy
        if self._calibrating:
            elapsed = time.time() - self._calib_start_time
            if elapsed < 3.0:
                pct = int((elapsed / 3.0) * 100)
                self.calib_bar.setValue(pct)
                self.calib_status_lbl.setText(f"Đang đo: {3.0 - elapsed:.1f}s ({len(self._calib_samples)} mẫu)")
            else:
                self._calibrating = False
                self.btn_calib.setEnabled(True)
                self.calib_bar.setValue(100)
                if len(self._calib_samples) >= 10:
                    ear_avg = float(np.mean(self._calib_samples))
                    new_close = float(np.clip(ear_avg * 0.70, 0.14, 0.30))
                    new_open = float(np.clip(ear_avg * 0.85, new_close + 0.02, 0.35))
                    self.slider_ear.setValue(int(new_close * 100))
                    self.fusion.set_thresholds(ear_close=new_close, ear_open=new_open)
                    self.calib_status_lbl.setText(f"✅ Xong! EAR mở: {ear_avg:.2f} ➔ Ngưỡng: {new_close:.2f}")
                    self.calib_status_lbl.setStyleSheet("color: #2E7D32; font-size: 11px; font-weight: bold;")
                else:
                    self.calib_status_lbl.setText("⚠️ Không đủ mẫu, vui lòng thử lại!")
                    self.calib_status_lbl.setStyleSheet("color: #C62828; font-size: 11px; font-weight: bold;")

        # 2. Cập nhật Frame lên Video Label
        with self._lock:
            frame = None if self._display_bgr is None else self._display_bgr.copy()
            st = self._state
            ear, mar, pitch, yaw = self._ear, self._mar, self._pitch, self._yaw
            fps, face_ok, dt = self._fps, self._face_ok, self._dt
            mean_bright, is_too_dark = self._mean_brightness, self._is_too_dark

        if frame is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(qimg.scaled(640, 480, Qt.KeepAspectRatio)))

        # 3. Hiển thị thông báo trạng thái
        color = {
            AlertLevel.NONE: "#2E7D32",
            AlertLevel.EARLY: "#E6A700",
            AlertLevel.FATIGUE: "#E65100",
            AlertLevel.STRONG: "#C62828",
        }[st.alert]

        if st.alert != AlertLevel.NONE:
            alert_html = f"<p style='color:{color};font-weight:bold;font-size:14px;'>⚠️ {st.alert_text}</p>"
        elif face_ok:
            alert_html = "<p style='color:#2E7D32;font-weight:bold;font-size:13px;'>Tỉnh Táo — Không Cảnh Báo</p>"
        elif is_too_dark:
            alert_html = "<p style='color:#C62828;font-weight:bold;font-size:13px;'>⚠️ Cabin Quá Tối — Đang Kích Hoạt Cân Sáng</p>"
        else:
            alert_html = "<p style='color:#888;font-size:13px;'>Chưa phát hiện khuôn mặt</p>"

        # Đánh giá môi trường chiếu sáng
        light_txt = (
            "<span style='color:#C62828;'>Quá Tối</span>"
            if is_too_dark
            else "<span style='color:#2E7D32;'>Đủ Sáng</span>"
        )

        svm_txt = "chưa train" if st.svm_drowsy is None else ("buồn ngủ" if st.svm_drowsy else "tỉnh")

        self.info_label.setText(
            f"<div style='font-family:Arial;color:#2C3E50;line-height:1.4;'>"
            f"<h3 style='margin:0 0 6px 0;text-align:center;color:{color};'>HỆ THỐNG GIÁM SÁT TÀI XẾ</h3>"
            f"<hr style='border:1px solid {color};margin:4px 0 8px 0;'>"
            f"{alert_html}"
            f"<p><b>Ánh sáng cabin:</b> {mean_bright:.0f}/255 ({light_txt})</p>"
            f"<p><b>FPS / Độ trễ:</b> {fps:.1f} FPS ({dt*1000:.0f} ms)</p>"
            f"<p><b>EAR:</b> {ear:.3f} &nbsp;|&nbsp; <b>MAR:</b> {mar:.3f}</p>"
            f"<p><b>Góc đầu (Pitch / Yaw):</b> {pitch:.1f}° / {yaw:.1f}°</p>"
            f"<p><b>👁️ Chớp mắt:</b> {st.blinks} &nbsp;({st.blink_rate:.0f} lần/phút)</p>"
            f"<p><b>💤 Microsleep:</b> {st.microsleep_s:.2f}s (ngưỡng: {self.fusion.microsleep_th:.1f}s)</p>"
            f"<p><b>😮 Ngáp:</b> {st.yawns} lần &nbsp;|&nbsp; <b>Thời lượng:</b> {st.yawn_duration_s:.1f}s</p>"
            f"<p><b>PERCLOS (60s):</b> {st.perclos_60*100:.1f}% (ngưỡng: 30%)</p>"
            f"<p><b>Nhịp gật đầu:</b> {st.nod_count}</p>"
            f"<p><b>Mô hình SVM/RF:</b> {svm_txt}</p>"
            f"<hr style='border:1px solid #E0E0E0;margin:6px 0;'>"
            f"<p style='font-size:10px;color:#888;margin:0;'>Nhật ký: {self.logger.path}</p>"
            f"</div>"
        )

    def closeEvent(self, event):
        self.stop_event.set()
        self.ui_timer.stop()
        if hasattr(self, "capture_thread") and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=0.3)
        if hasattr(self, "process_thread") and self.process_thread.is_alive():
            self.process_thread.join(timeout=0.3)
        self.logger.close()
        self.cap.release()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DrowsinessDetector()
    window.show()
    sys.exit(app.exec_())
