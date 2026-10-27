"""Lớp A — tiền xử lý ảnh (cân sáng thích ứng, khử nhiễu bảo toàn cạnh, đánh giá ánh sáng)."""

from __future__ import annotations

import cv2
import numpy as np


def analyze_lighting(bgr: np.ndarray) -> dict[str, float | bool]:
    """Phân tích độ rọi/ánh sáng cabin lái xe.
    
    Trả về:
        mean_brightness: Giá trị độ sáng trung bình [0, 255].
        is_too_dark: True nếu cabin quá tối (< 35/255).
        is_too_bright: True nếu cabin bị chói lóa (> 220/255).
    """
    if bgr is None or bgr.size == 0:
        return {"mean_brightness": 0.0, "is_too_dark": True, "is_too_bright": False}
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    mean_val = float(np.mean(gray))
    return {
        "mean_brightness": mean_val,
        "is_too_dark": mean_val < 35.0,
        "is_too_bright": mean_val > 220.0,
    }


def gray_world_white_balance(bgr: np.ndarray) -> np.ndarray:
    """Cân trắng Gray-World: đưa trung bình R,G,B về cùng mức xám."""
    img = bgr.astype(np.float32)
    means = img.reshape(-1, 3).mean(axis=0)
    gray = float(means.mean())
    scale = gray / (means + 1e-6)
    balanced = img * scale
    return np.clip(balanced, 0, 255).astype(np.uint8)


def adaptive_gamma_correction(bgr: np.ndarray, target_mid: float = 0.5) -> tuple[np.ndarray, float]:
    """Hiệu chỉnh Gamma thích ứng (Adaptive Gamma Correction):
    
    Tự động tính độ sáng trung bình m = mean / 255:
    - Nếu ảnh tối (m < 0.45): gamma < 1.0 (kéo sáng vùng tối theo hàm lũy thừa).
    - Nếu ảnh chói (m > 0.65): gamma > 1.0 (nén vùng chói).
    - Dùng bảng tra LUT (Lookup Table) 256 giá trị giúp tốc độ xử lý đạt < 1ms (Real-time).
    """
    if bgr is None or bgr.size == 0:
        return bgr, 1.0
    
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    mean_val = float(np.mean(gray))
    norm_mean = max(0.02, min(0.98, mean_val / 255.0))
    
    # Tính gamma thích ứng
    gamma = np.log(target_mid) / np.log(norm_mean)
    gamma = float(np.clip(gamma, 0.4, 1.8))
    
    # Tạo LUT tăng tốc
    lut = np.array([((i / 255.0) ** gamma) * 255.0 for i in range(256)]).astype(np.uint8)
    corrected = cv2.LUT(bgr, lut)
    return corrected, gamma


def bilateral_denoise(bgr: np.ndarray, d: int = 5, sigma_color: float = 30.0, sigma_space: float = 30.0) -> np.ndarray:
    """Khử nhiễu bảo toàn cạnh (Edge-Preserving Bilateral Filter):
    
    Thay thế cho Gaussian Blur đơn thuần: làm mịn hạt nhiễu do ISO cao trong bóng tối
    nhưng giữ nguyên độ sắc nét của viền mi mắt (eye contour) và khóe môi.
    """
    return cv2.bilateralFilter(bgr, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)


def enhance_lighting(
    bgr: np.ndarray,
    clip_limit: float = 2.0,
    use_adaptive_gamma: bool = True,
    use_bilateral: bool = True,
) -> np.ndarray:
    """Quy trình tăng cường ảnh toàn diện cho cabin lái xe:
    1. Cân trắng Gray-World khử ám vàng/xanh.
    2. Adaptive Gamma Correction tự động cứu sáng khi lái xe ban đêm.
    3. CLAHE trên kênh L (LAB) cân bằng tương phản cục bộ không phá vỡ màu sắc.
    4. Bilateral Filter khử nhiễu cảm biến ban đêm mà bảo toàn biên mi mắt.
    """
    if bgr is None or bgr.size == 0:
        return bgr

    # Bước 1: Cân trắng
    balanced = gray_world_white_balance(bgr)

    # Bước 2: Gamma thích ứng (nếu bật)
    if use_adaptive_gamma:
        gamma_img, _ = adaptive_gamma_correction(balanced, target_mid=0.5)
    else:
        gamma_img = balanced

    # Bước 3: CLAHE trên kênh L (không gian màu CIE LAB)
    lab = cv2.cvtColor(gamma_img, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l_eq = clahe.apply(l_ch)
    merged = cv2.merge((l_eq, a_ch, b_ch))
    enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    # Bước 4: Khử nhiễu bảo toàn cạnh
    if use_bilateral:
        return bilateral_denoise(enhanced, d=5, sigma_color=30.0, sigma_space=30.0)
    return cv2.GaussianBlur(enhanced, (3, 3), 0)


def to_rgb(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
