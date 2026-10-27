"""Bộ chạy thời gian thực trực quan bằng thuần OpenCV HighGUI.
Tích hợp:
- Cân sáng thích ứng Adaptive Gamma + Bilateral Filter bảo toàn biên mi mắt.
- Cơ chế Fallback khi cabin thiếu sáng hoặc mất dấu mặt.
- Âm thanh cảnh báo đa tầng theo AlertLevel.
- Bảng Trackbar điều chỉnh trực tiếp EAR, MAR, Microsleep ngay trên cửa sổ.
- Phím tắt:
    'c': Tự động cân chỉnh mắt (Auto-Calibration 3s)
    'm': Bật/Tắt âm thanh cảnh báo
    'g': Bật/Tắt Adaptive Gamma
    'b': Bật/Tắt Bilateral Filter
    'q' / ESC: Thoát
"""

from __future__ import annotations

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Khắc phục triệt để lỗi C++ Unicode (MediaPipe binarypb) khi đường dẫn chứa tiếng Việt có dấu:
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

import threading
import time
import cv2
import numpy as np
import mediapipe as mp

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
    if not audio_enabled or alert == AlertLevel.NONE:
        return
    try:
        import winsound
        if alert == AlertLevel.STRONG:
            winsound.Beep(1800, 140)
            winsound.Beep(2200, 160)
            winsound.Beep(1800, 140)
        elif alert == AlertLevel.FATIGUE:
            winsound.Beep(1300, 160)
            winsound.Beep(1300, 160)
        elif alert == AlertLevel.EARLY:
            winsound.Beep(900, 140)
    except Exception:
        print("\a", end="", flush=True)


def main():
    print("=" * 60)
    print("  HE THONG GIAM SAT TAI XE REAL-TIME (OPENCV)")
    print("  Phim tat:")
    print("    [c]: Tu dong can chinh mat (Auto-Calibration 3s)")
    print("    [m]: Bat/Tat am thanh canh bao")
    print("    [g]: Bat/Tat Adaptive Gamma")
    print("    [b]: Bat/Tat Bilateral Filter")
    print("    [q] hoac [ESC]: Thoat ung dung")
    print("=" * 60)

    # Khởi tạo camera với DirectShow (CAP_DSHOW) để tránh lỗi MSMF can't grab frame trên Windows
    cap = None
    for idx in (0, 1, 2):
        c = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if c.isOpened():
            c.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            c.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            ret, frame = c.read()
            if ret and frame is not None and frame.size > 0:
                print(f"[Camera] Da ket noi Webcam index {idx} qua DirectShow (CAP_DSHOW).")
                cap = c
                break
            c.release()

    if cap is None:
        print("[Loi] Khong the ket noi den bat ky Webcam nao.")
        return

    win_name = "DMS - Giam Sat Tai Xe Real-Time"
    cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)

    # Khởi tạo modules
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    geometry = FaceGeometry()
    smoother = TemporalSmoother()
    fusion = AlertFusion()
    deep = DeepDetectors()
    ml = TraditionalDrowsinessModel()
    logger = SessionLogger()

    # Trackbars điều chỉnh tham số trực tiếp
    def on_ear(val):
        close = max(0.12, val / 100.0)
        fusion.set_thresholds(ear_close=close, ear_open=close + 0.04)

    def on_mar(val):
        yawn = max(0.35, val / 100.0)
        fusion.set_thresholds(mar_yawn=yawn, mar_rest=max(0.20, yawn - 0.15))

    def on_micro(val):
        fusion.set_thresholds(microsleep_s=max(0.5, val / 10.0))

    cv2.createTrackbar("EAR Nham x100", win_name, 21, 32, on_ear)
    cv2.createTrackbar("MAR Ngap x100", win_name, 55, 80, on_mar)
    cv2.createTrackbar("Micro x10s", win_name, 10, 25, on_micro)

    audio_enabled = True
    use_gamma = True
    use_bilateral = True

    # Calibration state
    calibrating = False
    calib_t0 = 0.0
    calib_samples = []

    t_prev = time.time()
    prev_blink_rate = 0.0
    prev_perclos = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        now = time.time()
        dt = max(now - t_prev, 1e-3)
        t_prev = now
        fps = 1.0 / dt

        # 1. Phân tích độ sáng
        light_info = analyze_lighting(frame)
        is_too_dark = bool(light_info["is_too_dark"])
        mean_bright = float(light_info["mean_brightness"])

        # 2. Tiền xử lý ảnh
        enhanced = enhance_lighting(
            frame,
            clip_limit=2.0,
            use_adaptive_gamma=use_gamma,
            use_bilateral=use_bilateral,
        )

        # 3. MediaPipe Face Mesh
        rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
        mesh = face_mesh.process(rgb)
        ih, iw = enhanced.shape[:2]

        geo = None
        face_ok = False
        votes = []
        svm_flag = None

        if mesh.multi_face_landmarks:
            face_ok = True
            geo = geometry.compute(mesh.multi_face_landmarks[0], iw, ih)
            ear, mar, pitch, yaw = smoother.update(geo.ear, geo.mar, geo.pitch, geo.yaw)

            if calibrating:
                calib_samples.append(ear)

            left_roi = crop_with_padding(enhanced, geo.left_eye_pts)
            right_roi = crop_with_padding(enhanced, geo.right_eye_pts)
            mouth_roi = crop_with_padding(enhanced, geo.mouth_pts)

            eye_vote = deep.vote_eyes(left_roi, right_roi)
            yawn_vote = deep.vote_yawn(mouth_roi)
            votes = [eye_vote, yawn_vote]
            svm_flag, _ = ml.predict(ear, mar, pitch, prev_blink_rate, prev_perclos)
        else:
            ear = mar = pitch = yaw = 0.0

        # 4. Fusion cảnh báo
        state = fusion.update(
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
        prev_perclos = state.perclos_60
        prev_blink_rate = state.blink_rate

        # 5. Vẽ overlay kết quả
        vis = draw_overlay(enhanced, geo, state)

        # Thanh trạng thái phía trên
        status_line = (
            f"FPS: {fps:.1f} | Sang: {mean_bright:.0f}/255 ({'TOI' if is_too_dark else 'OK'}) | "
            f"Am thanh: {'BAT' if audio_enabled else 'TAT'} [M] | Gamma: {'BAT' if use_gamma else 'TAT'} [G]"
        )
        cv2.putText(vis, status_line, (10, vis.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        # Xử lý Auto-Calibration
        if calibrating:
            elapsed = now - calib_t0
            if elapsed < 3.0:
                cv2.putText(
                    vis,
                    f"DANG CAN CHINH: {3.0 - elapsed:.1f}s ({len(calib_samples)} mau) - NHIN THANG!",
                    (20, 180),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 165, 255),
                    2,
                )
            else:
                calibrating = False
                if len(calib_samples) >= 10:
                    ear_avg = float(np.mean(calib_samples))
                    new_close = float(np.clip(ear_avg * 0.70, 0.14, 0.30))
                    cv2.setTrackbarPos("EAR Nham x100", win_name, int(new_close * 100))
                    print(f"[Calibration] Thanh cong! EAR mo tb={ear_avg:.3f} -> Nguong nham={new_close:.3f}")
                else:
                    print("[Calibration] That bai: Khong du mau mat.")

        # 6. Âm thanh cảnh báo
        if fusion.should_beep(now):
            threading.Thread(target=play_alert_sound, args=(state.alert, audio_enabled), daemon=True).start()

        # 7. Ghi log
        logger.write({
            "timestamp": f"{now:.3f}",
            "fps": f"{fps:.1f}",
            "ear": f"{ear:.4f}",
            "mar": f"{mar:.4f}",
            "alert": state.alert.value,
        })

        cv2.imshow(win_name, vis)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q'), ord('Q')):
            break
        elif key in (ord('c'), ord('C')):
            calibrating = True
            calib_t0 = time.time()
            calib_samples.clear()
            print("[Calibration] Bat dau do mat trong 3 giay...")
        elif key in (ord('m'), ord('M')):
            audio_enabled = not audio_enabled
            print(f"[Audio] Am thanh: {'BAT' if audio_enabled else 'TAT'}")
        elif key in (ord('g'), ord('G')):
            use_gamma = not use_gamma
            print(f"[Preprocessing] Adaptive Gamma: {'BAT' if use_gamma else 'TAT'}")
        elif key in (ord('b'), ord('B')):
            use_bilateral = not use_bilateral
            print(f"[Preprocessing] Bilateral Filter: {'BAT' if use_bilateral else 'TAT'}")

    cap.release()
    cv2.destroyAllWindows()
    logger.close()


if __name__ == "__main__":
    main()
