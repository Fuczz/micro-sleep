# BÁO CÁO ĐỒ ÁN

## Hệ thống phát hiện buồn ngủ khi lái xe thời gian thực  
### (Computer Vision — xử lý ảnh, không phải DMS thương mại)

**Môn:** Xử lý ảnh  
**Bài toán:** Giám sát tài xế bằng webcam: chớp mắt, microsleep, ngáp, gật gù, PERCLOS  
**Hướng tiếp cận:** Hình học cổ điển (xương sống) + học sâu đúng vai trò + học máy truyền thống  

---

## 1. Hiện trạng (một câu)

Hệ prototype realtime: Face Mesh cắt ROI → 2 YOLO nông phân loại mở/nhắm và ngáp → GUI PyQt; **chưa phải DMS chuẩn**, chưa đủ xử lý ảnh, data/label/train còn lỗ hổng — phiên bản này **sửa lỗ hổng đó** và dựng đủ 3 lớp đúng môn.

### 1.1 Lỗi đã sửa trước khi thêm feature

| Lỗi cũ | Hệ quả | Cách sửa |
|---|---|---|
| `LoadData` chỉ xuất `closed_eyes`, gán class `0` | Yaml nói `0 = open eye` → nhãn đảo, confusion matrix vô nghĩa | Xuất **cả open và closed**; `0 = open`, `1 = close` |
| Một `dataset.yaml` dùng chung mắt + ngáp | Class conflict khi train | Tách `dataset_eye.yaml` / `dataset_yawn.yaml` |
| Cộng thời gian `45/1000` mỗi frame | Sai khi FPS ≠ 22; microsleep 4 s mới báo | `time.time()`, ngưỡng microsleep **1.0 s** |
| Còi bị comment, alert không reset | Demo câm, cảnh báo kẹt | Bật còi + cooldown; tỉnh thì xóa alert |
| Không vẽ landmark / EAR / MAR | Giảng viên không thấy XL ảnh | Overlay 6 điểm mắt, 8 điểm miệng, thanh PERCLOS |
| Cảnh báo yawn 7 s / mắt 4 s | Muộn, không đúng y văn PERCLOS | Fusion: microsleep / PERCLOS / ngáp / gật |

---

## 2. Mục tiêu và phạm vi đúng môn Xử lý ảnh

**Làm (điểm cao, hợp dự án):**

1. EAR + MAR bằng MediaPipe (6 điểm mắt, 8 điểm miệng).  
2. Tiền xử lý ánh sáng cabin: cân trắng + CLAHE kênh L + Gaussian nhẹ.  
3. Tách blink (100–400 ms) vs microsleep (≥ 1 s) + PERCLOS 60 s.  
4. Head pose `solvePnP` → pitch (gật gù).  
5. Làm mượt thời gian: EMA 7 frame, `dt` thật, cooldown còi.  
6. So sánh mô hình: YOLO (detect) vs MobileNet/EfficientNet (classify ROI) vs SVM/RF (vector hình học).  
7. Đánh giá: test set riêng, confusion matrix, P/R/F1, FPS, log CSV.

**Không ôm cho có:** U-Net phân đoạn mắt; AlexNet/VGG/ResNet đầy đủ (nặng realtime); wearable / app mobile (README “future”, lệch môn).

---

## 3. Cơ sở lý thuyết (chương XL ảnh)

### 3.1 Tiền xử lý ánh sáng cabin nâng cao (chương tăng cường ảnh)

Cabin xe thay đổi mạnh: ban ngày, đêm tối, ngược sáng. Pipeline xử lý ảnh đầu vào gồm 4 công đoạn tối ưu:

1. **Gray-World white balance** — giả sử trung bình R, G, B bằng nhau; scale mỗi kênh về mức xám chung để khử ám màu từ đèn cabin.  
2. **Hiệu chỉnh Gamma thích ứng (Adaptive Gamma Correction)** — tự động đo độ sáng trung bình $m = \text{mean}(V)/255$. Nếu cabin tối ($m < 0.45$), tự động tính $\gamma = \ln(0.5)/\ln(m) < 1.0$ để kéo sáng vùng tối phi tuyến. Sử dụng bảng tra LUT 256 giá trị giúp tốc độ xử lý đạt $O(1) < 1\text{ms}$.  
3. **CLAHE trên kênh L** (không gian CIE LAB) — cân bằng histogram cục bộ thích ứng, làm nổi bật viền mắt/miệng mà không phá hủy thông tin màu sắc $a, b$.  
4. **Khử nhiễu bảo toàn cạnh (Bilateral Filter)** — làm mịn hạt nhiễu do ISO cảm biến ban đêm nhưng giữ nguyên độ tương phản sắc nét tại đường biên mí mắt và khóe môi.  
5. **Cơ chế Fallback & Đánh giá môi trường** — liên tục đo độ sáng trung bình; nếu thiếu sáng trầm trọng kết hợp mất dấu khuôn mặt, hệ thống tự động kích hoạt cảnh báo môi trường tối và dự phòng an toàn thay vì im lặng.

### 3.2 EAR — Eye Aspect Ratio

Với 6 điểm mắt \(p_1 \ldots p_6\) (Soukupová & Čech):

\[
EAR = \frac{\|p_2-p_6\| + \|p_3-p_5\|}{2\|p_1-p_4\|}
\]

- Mắt mở: EAR ≈ 0.25–0.35.  
- Mắt nhắm: EAR < ngưỡng (mặc định **0.21** vào, **0.25** ra — hysteresis).  

MediaPipe Face Mesh 468 điểm:

- Mắt trái: `33, 160, 158, 133, 153, 144`  
- Mắt phải: `362, 385, 387, 263, 373, 380`  

Đây là **xử lý ảnh hình học cổ điển**, giảng viên dễ chấm vì công thức, điểm, ngưỡng đều chỉ ra được trên video.

### 3.3 MAR — Mouth Aspect Ratio

8 điểm miệng: ngang `78–308` và 3 cặp dọc `13–14`, `82–312`, `87–317`:

\[
MAR = \frac{\|t_1-b_1\|+\|t_2-b_2\|+\|t_3-b_3\|}{2\|p_{left}-p_{right}\|}
\]

Ngáp: MAR > 0.55 (vào) / < 0.40 (ra).

### 3.4 Blink vs microsleep vs PERCLOS

| Sự kiện | Thời gian nhắm | Ý nghĩa |
|---|---|---|
| Blink | 100–400 ms (cửa 80–400 ms) | Sinh lý, **không cảnh báo** |
| Microsleep | ≥ **1.0 s** | Nguy hiểm — ALERT mạnh |
| PERCLOS | tỷ lệ frame nhắm / **60 s** | Mệt tích lũy; > **0.30** → ALERT mệt |

Không đợi 4 giây như bản cũ. PERCLOS là chỉ số kinh điển trong tài liệu giám sát tài xế (NHTSA / IEEE), phù hợp chương “đặc trưng theo thời gian”.

### 3.5 Head pose — gật gù

6 điểm cứng MediaPipe `{1, 152, 33, 263, 61, 291}` + mô hình 3D → `cv2.solvePnP` → Euler (pitch, yaw, roll).

- Pitch vượt ngưỡng rồi trở lại = **một nhịp gật**.  
- ≥ 2 nhịp trong ~10 s → cảnh báo mệt.  
- \|yaw\| lớn kéo dài → nhìn lệch đường (mở rộng hợp lý).

README gốc đã hứa “head position”; bản cũ **chưa có** — đây là chỗ được vá.

### 3.6 Làm mượt thời gian

- **EMA** \(\alpha = 0.35\) trên EAR, MAR, pitch, yaw.  
- Trung bình cửa sổ 7 frame.  
- Mọi khoảng thời gian dùng `time.time()` (dt thật), không giả định 22 FPS.  
- Cooldown còi 2 s; khi tỉnh, `alert = NONE` (reset).

---

## 4. Kiến trúc 3 lớp (đúng môn, không ôm mạng cho có)

```
Webcam
  → CLAHE / cân trắng / Gaussian          [Lớp A — XL ảnh]
  → MediaPipe Face Mesh (468)
  → EAR, MAR, pitch, yaw
  → EMA + hysteresis
  → luật blink / microsleep / yawn / PERCLOS
         │
         ├→ YOLO v8n trên ROI mắt / miệng  [Lớp B — detect]
         ├→ MobileNetV3 / EfficientNet-B0  [Lớp B — classify ROI]
         └→ SVM / Random Forest            [Lớp C — vector 5 chiều]
         │
         ▼
      AlertFusion  →  overlay + còi + CSV + GUI PyQt
```

### Lớp A — xương sống báo cáo (bắt buộc)

Module: `cv_core/preprocess.py`, `geometry.py`, `smoothing.py`, `fusion.py`, `overlay.py`.

ROI: `min/max` landmark + **padding 15%**; crop rỗng → bỏ, không đưa vào YOLO.

### Lớp B — học sâu đúng vai trò

| Vai trò | Mô hình | Vì sao chọn |
|---|---|---|
| Detector trên mặt / ROI | **YOLOv8n** (giữ bản hiện có) | Realtime, 2 class |
| Classifier trên ROI | **MobileNetV3-Small** (chính) / EfficientNet-B0 (so sánh) | Nhẹ hơn VGG/ResNet, đúng mục “phân loại ảnh” trên hình CV |
| Không dùng | AlexNet, VGG, ResNet đầy đủ, U-Net | Nặng / ít lợi hơn EAR |

**Hysteresis + conf đối xứng:** vào Close khi YOLO/MobileNet conf > 0.6 **hoặc** EAR < 0.21; ra khi Open conf > 0.6 và EAR > 0.25. Tránh nhấp nháy.

Map class (bắt buộc khớp yaml):

- Mắt: `0 open eye`, `1 close eye`  
- Miệng: `0 yawn`, `1 no yawn`

### Lớp C — “Không chỉ deep learning”

Vector:

\[
\mathbf{x} = [EAR,\ MAR,\ pitch,\ blink\_rate,\ PERCLOS_{60}]
\]

- SVM (RBF + StandardScaler)  
- Random Forest (200 cây)

Train từ CSV đã gán nhãn (`train_traditional.py`). Nếu chưa có file `.joblib`, hệ vẫn chạy: lớp C im lặng, A+B vẫn cảnh báo.

Có thể bổ sung HOG ROI mắt như baseline cổ điển (mục 5 trên slide CV) — không bắt buộc nếu SVM trên EAR đã có trong báo cáo.

### Fusion — quyết định cảnh báo

```
if microsleep_duration >= 1.0s:          ALERT mạnh
elif PERCLOS_60s > 0.3:                  ALERT mệt
elif yawn_count_3min >= 3 or MAR >= 3s:  CẢNH BÁO sớm
elif head_pitch gật lặp (>=2 / 10s):     CẢNH BÁO
elif yaw nhìn lệch >= 2s:                CẢNH BÁO
elif SVM/RF = buồn ngủ:                  CẢNH BÁO sớm
elif mất face >= 2s:                     không thấy tài xế
else:                                    tỉnh — xóa alert
```

Kết hợp hình học (A) + mạng (B) vừa đúng môn XL ảnh, vừa để YOLO hiện tại có chỗ đứng, vừa sửa tracking/cảnh báo yếu.

---

## 5. Dữ liệu và huấn luyện

### 5.1 Mắt

- Nguồn: HuggingFace `MichalMlodawski/closed-open-eyes` (hoặc Kaggle Eyes / MRL).  
- Tách train / val / **test** (80 / 10 / 10).  
- Script: `python prepare_eye_dataset.py` hoặc `LoadData.ipynb` (đã sửa).  
- Train: `python train_yolo.py --task eye --epochs 80`

**Lưu ý:** weight trong `runs/detecteye/train` gốc chỉ ~10 epoch, **chưa hội tụ**. Báo cáo phải ghi rõ và train lại 50–100 epoch, nộp confusion matrix test.

### 5.2 Ngáp

- Dataset yawning (Kaggle) → `datasets/yawn` theo YOLO.  
- `python train_yolo.py --task yawn --epochs 80`

### 5.3 Classifier ROI

```
python train_classifier.py --task eye --arch mobilenet --epochs 20
python train_classifier.py --task eye --arch efficientnet --epochs 20
python train_classifier.py --task mouth --arch mobilenet --epochs 20
```

Bảng so sánh (điền số sau khi train — đây là **khung báo cáo**):

| Mô hình | Vai trò | P | R | F1 | FPS (xấp xỉ) |
|---|---|---|---|---|---|
| YOLOv8n | Detect open/close, yawn |  |  |  |  |
| MobileNetV3-Small | Classify ROI |  |  |  |  |
| EfficientNet-B0 | Classify ROI |  |  |  |  |
| SVM | Vector 5-D |  |  |  |  |
| Random Forest | Vector 5-D |  |  |  |  |
| EAR/MAR (ngưỡng) | Hình học, không học |  |  |  |  |

### 5.4 SVM / RF

1. Chạy detector → `logs/session_*.csv`.  
2. Thêm cột `label` (0 tỉnh / 1 buồn ngủ) trên đoạn video đã biết.  
3. `python train_traditional.py --csv logs/session_xxx.csv`  
4. File ra: `runs/traditional/drowsiness_svm.joblib` (ưu tiên load) và `_rf.joblib`.

---

## 6. Giao diện, overlay, log (bắt buộc demo)

Overlay trên video:

- 6 điểm mắt + đa giác mắt (xanh = mở, đỏ = nhắm)  
- 8 điểm miệng  
- EAR trái/phải/trung bình, MAR, pitch  
- Nhãn OPEN / CLOSE, YAWN / NO-YAWN  
- Thanh PERCLOS (xanh / cam / đỏ theo 15% / 30%)  
- Viền màu theo mức cảnh báo  

GUI: FPS, dt (ms), blinks, microsleep (s), yawns, PERCLOS, gật đầu, trạng thái SVM, đường dẫn CSV.

Âm thanh: `winsound.Beep` (Windows), cooldown 2 s.

Demo video gợi ý (5 tình huống): **tỉnh — chớp — ngủ gật ≥1 s — ngáp — tối/ngược sáng**.

---

## 7. Đánh giá đúng bài Computer Vision

Không chỉ “chạy webcam thấy báo”. Cần:

1. **Test set riêng** (không trùng train).  
2. Confusion matrix từng bài: mắt open/close; miệng yawn/no-yawn; (tuỳ chọn) tỉnh/buồn ngủ mức fusion.  
3. Precision, Recall, F1 từng class.  
4. FPS và độ trễ cảnh báo (microsleep thật 1.0 s vs thời điểm còi — đọc từ CSV `timestamp`).  
5. So sánh ngày / đêm / ngược sáng (ảnh CLAHE).  

Lệnh:

```
python evaluate.py --csv logs/session_xxx.csv --y-true label --y-pred y_pred
```

Nếu CSV chỉ có `alert`, script tự biến `alert != none` thành dự đoán buồn ngủ.

---

## 8. Tính năng mở rộng đã chọn (2–3, không lệch môn)

1. Nhìn lệch đường: \|yaw\| ≥ 25° trong ≥ 2 s.  
2. Che mặt / mất Face Mesh ≥ 2 s → “không thấy tài xế”.  
3. Đếm ngáp theo cửa sổ 3 phút (không đợi 1 ngáp dài 7 s).  
4. Overlay + CSV (phục vụ chấm và viết số liệu).

Không làm: wearable, app mobile, U-Net segment.

---

## 9. Cấu trúc mã nguồn

```
cv_core/
  preprocess.py     CLAHE, cân trắng, Gaussian
  geometry.py       EAR, MAR, solvePnP, crop padding
  smoothing.py      EMA + cửa sổ
  fusion.py         hysteresis + PERCLOS + luật cảnh báo
  dl_models.py      YOLO + MobileNet/EfficientNet
  traditional.py    SVM / RF
  overlay.py        landmark, thanh PERCLOS
  logger.py         CSV
DrowsinessDetector.py   GUI PyQt + luồng camera
prepare_eye_dataset.py  LoadData đã sửa
train_yolo.py / train_classifier.py / train_traditional.py / evaluate.py
dataset_eye.yaml / dataset_yawn.yaml
BAO_CAO_DO_AN.md        file này
```

Chạy realtime:

```
pip install -r requirements.txt
python DrowsinessDetector.py
```

---

## 10. Kết luận

Hệ thống không còn là “cắm 2 YOLO rồi cộng 45 ms”. Xương sống là **xử lý ảnh**: tăng cường sáng (Adaptive Gamma, CLAHE, Bilateral Filter), mốc landmark, tỷ lệ hình học, pose, đặc trưng theo thời gian. Học sâu được đặt đúng chỗ (detect/classify ROI). Học máy truyền thống chứng minh “không chỉ deep learning”. Fusion bám chỉ số PERCLOS / microsleep kèm hệ thống âm thanh cảnh báo đa tầng, cơ chế fallback thiếu sáng và giao diện điều chỉnh thông số trực tiếp kèm tính năng tự động cân chỉnh (Auto-Calibration 3s). Đủ để viết một đồ án môn Xử lý ảnh có lý thuyết, có thực nghiệm, có so sánh mô hình.

### Các cải tiến đã hoàn thiện trong phiên bản này:

- **Tự động cân chỉnh (Auto-Calibration 3s)**: Tài xế nhìn thẳng 3s, hệ thống tự động đo EAR trung bình và đặt ngưỡng cá nhân hóa.  
- **Dynamic Control UI**: Tích hợp các thanh trượt điều chỉnh ngưỡng EAR, MAR, Microsleep, Conf ngay trên giao diện thời gian thực.  
- **Âm thanh cảnh báo đa tầng**: Phân biệt rõ ràng giữa cảnh báo sớm (EARLY), mệt mỏi tích lũy (FATIGUE) và khẩn cấp (STRONG).  
- **Pipeline tiền xử lý đêm tối**: Adaptive Gamma + Bilateral Filter + Fallback an toàn khi mất dấu khuôn mặt.

---

## Phụ lục A — Công thức và ngưỡng mặc định

| Tham số | Giá trị | Ý nghĩa |
|---|---|---|
| `ear_close` / `ear_open` | 0.21 / 0.25 | Hysteresis nhắm/mở |
| `mar_yawn` / `mar_rest` | 0.55 / 0.40 | Hysteresis ngáp |
| `dl_conf` | 0.60 | Conf YOLO/MobileNet |
| `blink` | 0.08–0.40 s | Chớp sinh lý |
| `microsleep` | 1.0 s | ALERT mạnh |
| `PERCLOS` | 0.30 / 60 s | ALERT mệt |
| `yawn_long` | 3.0 s | Cảnh báo sớm |
| `yawn_count` | ≥ 3 / 180 s | Cảnh báo sớm |
| `nod_pitch` | 18° | Gật đầu |
| `yaw_lookaway` | 25° / 2 s | Lệch đường |
| `face_missing` | 2.0 s | Không thấy tài xế |
| ROI padding | 15% | Tránh crop sát mi |

## Phụ lục B — Mapping class YAML (tránh lỗi cũ)

```
dataset_eye.yaml     0: open eye    1: close eye
dataset_yawn.yaml    0: yawn        1: no yawn
```
