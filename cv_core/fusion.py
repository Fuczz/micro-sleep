"""Hysteresis + fusion cảnh báo (hình học + deep + truyền thống)."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum


class AlertLevel(str, Enum):
    NONE = "none"
    EARLY = "early"
    FATIGUE = "fatigue"
    STRONG = "strong"


@dataclass
class DetectorVote:
    eye_closed: bool | None = None
    eye_conf: float = 0.0
    yawn: bool | None = None
    yawn_conf: float = 0.0


@dataclass
class FusionState:
    blinks: int = 0
    yawns: int = 0
    microsleep_s: float = 0.0
    yawn_duration_s: float = 0.0
    perclos_60: float = 0.0
    blink_rate: float = 0.0
    nod_count: int = 0
    eyes_closed: bool = False
    yawning: bool = False
    alert: AlertLevel = AlertLevel.NONE
    alert_text: str = ""
    svm_drowsy: bool | None = None


class AlertFusion:
    """Luật fusion:

    if microsleep >= 1.0s: ALERT mạnh
    elif PERCLOS_60s > 0.3: ALERT mệt
    elif yawn_count_3min >= 3 or MAR kéo dài >= 3s: CẢNH BÁO sớm
    elif head_pitch gật lặp: CẢNH BÁO
    """

    def __init__(
        self,
        ear_close: float = 0.21,
        ear_open: float = 0.25,
        mar_yawn: float = 0.55,
        mar_rest: float = 0.40,
        dl_conf: float = 0.60,
        microsleep_s: float = 1.0,
        blink_max_s: float = 0.40,
        blink_min_s: float = 0.08,
        perclos_th: float = 0.30,
        yawn_long_s: float = 3.0,
        nod_pitch_th: float = 18.0,
        yaw_lookaway_th: float = 25.0,
        face_missing_s: float = 2.0,
        lookaway_s: float = 2.0,
    ):
        self.ear_close = ear_close
        self.ear_open = ear_open
        self.mar_yawn = mar_yawn
        self.mar_rest = mar_rest
        self.dl_conf = dl_conf
        self.microsleep_th = microsleep_s
        self.blink_max_s = blink_max_s
        self.blink_min_s = blink_min_s
        self.perclos_th = perclos_th
        self.yawn_long_s = yawn_long_s
        self.nod_pitch_th = nod_pitch_th
        self.yaw_lookaway_th = yaw_lookaway_th
        self.face_missing_s = face_missing_s
        self.lookaway_s = lookaway_s

        self.eyes_closed = False
        self.yawning = False
        self.close_t0 = None
        self.yawn_t0 = None
        self.blinks = 0
        self.yawns = 0
        self.nods = 0
        self._nod_down = False
        self._last_alert = AlertLevel.NONE
        self._alert_since = 0.0
        self._last_beep = 0.0
        self._last_face = None
        self._lookaway_t0 = None

        self.closed_events: deque[tuple[float, bool]] = deque()
        self.blink_times: deque[float] = deque()
        self.yawn_times: deque[float] = deque()
        self.nod_times: deque[float] = deque()

    def _hysteresis_closed(self, ear: float, votes: list[DetectorVote]) -> bool:
        dl_close = any(v.eye_closed is True and v.eye_conf >= self.dl_conf for v in votes)
        dl_open = any(v.eye_closed is False and v.eye_conf >= self.dl_conf for v in votes)
        if not self.eyes_closed:
            return ear < self.ear_close or dl_close
        # Ra trạng thái nhắm chỉ khi EAR mở và không còn DL khẳng định nhắm
        if ear > self.ear_open and dl_open and not dl_close:
            return False
        if ear > self.ear_open and not dl_close:
            return False
        return True

    def _hysteresis_yawn(self, mar: float, votes: list[DetectorVote]) -> bool:
        dl_yawn = any(v.yawn is True and v.yawn_conf >= self.dl_conf for v in votes)
        dl_no = any(v.yawn is False and v.yawn_conf >= self.dl_conf for v in votes)
        if not self.yawning:
            return mar > self.mar_yawn or dl_yawn
        if mar < self.mar_rest and dl_no and not dl_yawn:
            return False
        if mar < self.mar_rest and not dl_yawn:
            return False
        return True

    def set_thresholds(
        self,
        ear_close: float | None = None,
        ear_open: float | None = None,
        mar_yawn: float | None = None,
        mar_rest: float | None = None,
        microsleep_s: float | None = None,
        dl_conf: float | None = None,
    ):
        """Cập nhật các ngưỡng trực tiếp từ giao diện điều khiển."""
        if ear_close is not None:
            self.ear_close = ear_close
        if ear_open is not None:
            self.ear_open = ear_open
        if mar_yawn is not None:
            self.mar_yawn = mar_yawn
        if mar_rest is not None:
            self.mar_rest = mar_rest
        if microsleep_s is not None:
            self.microsleep_th = microsleep_s
        if dl_conf is not None:
            self.dl_conf = dl_conf

    def update(
        self,
        now: float,
        ear: float,
        mar: float,
        pitch: float,
        votes: list[DetectorVote],
        svm_drowsy: bool | None = None,
        face_ok: bool = True,
        yaw: float = 0.0,
        is_too_dark: bool = False,
    ) -> FusionState:
        # Cơ chế Fallback thông minh khi mất dấu khuôn mặt (do bóng tối hoặc quay mặt)
        if not face_ok:
            self.eyes_closed = False
            self.yawning = False
            self.close_t0 = None
            self.yawn_t0 = None
            if self._last_face is None:
                self._last_face = now
            missing = now - self._last_face
            alert, text = AlertLevel.NONE, ""
            if missing >= self.face_missing_s:
                if missing >= 4.0:
                    alert = AlertLevel.STRONG
                    if is_too_dark:
                        text = "ALERT MANH: Mat dau tai xe qua 4s (Cabin qua toi!)"
                    else:
                        text = "ALERT MANH: Mat dau tai xe qua 4s (Che mat / Mat tap trung!)"
                else:
                    alert = AlertLevel.EARLY
                    if is_too_dark:
                        text = "CANH BAO: Thieu anh sang - Khong thay khuon mat"
                    else:
                        text = "CANH BAO: Khong thay tai xe (mat face / che mat)"
            self._last_alert = alert
            return FusionState(
                blinks=self.blinks,
                yawns=self.yawns,
                perclos_60=0.0,
                blink_rate=float(len(self.blink_times)),
                nod_count=self.nods,
                alert=alert,
                alert_text=text,
            )
        self._last_face = now

        was_closed = self.eyes_closed
        was_yawn = self.yawning
        self.eyes_closed = self._hysteresis_closed(ear, votes)
        self.yawning = self._hysteresis_yawn(mar, votes)

        microsleep_s = 0.0
        yawn_s = 0.0

        if self.eyes_closed and not was_closed:
            self.close_t0 = now
        if self.eyes_closed and self.close_t0 is not None:
            microsleep_s = now - self.close_t0
        if was_closed and not self.eyes_closed and self.close_t0 is not None:
            dur = now - self.close_t0
            if self.blink_min_s <= dur <= self.blink_max_s:
                self.blinks += 1
                self.blink_times.append(now)
            self.close_t0 = None
            microsleep_s = 0.0

        if self.yawning and not was_yawn:
            self.yawn_t0 = now
            self.yawns += 1
            self.yawn_times.append(now)
        if self.yawning and self.yawn_t0 is not None:
            yawn_s = now - self.yawn_t0
        if was_yawn and not self.yawning:
            self.yawn_t0 = None
            yawn_s = 0.0

        # Gật đầu: pitch vượt ngưỡng rồi trở lại
        if pitch > self.nod_pitch_th:
            self._nod_down = True
        elif self._nod_down and pitch < self.nod_pitch_th * 0.5:
            self._nod_down = False
            self.nods += 1
            self.nod_times.append(now)

        self.closed_events.append((now, self.eyes_closed))
        cutoff_60 = now - 60.0
        while self.closed_events and self.closed_events[0][0] < cutoff_60:
            self.closed_events.popleft()
        if self.closed_events:
            n_closed = sum(1 for _, c in self.closed_events if c)
            perclos = n_closed / len(self.closed_events)
        else:
            perclos = 0.0

        while self.blink_times and self.blink_times[0] < now - 60.0:
            self.blink_times.popleft()
        blink_rate = len(self.blink_times)  # chớp / phút

        cutoff_3m = now - 180.0
        while self.yawn_times and self.yawn_times[0] < cutoff_3m:
            self.yawn_times.popleft()
        yawn_3min = len(self.yawn_times)

        while self.nod_times and self.nod_times[0] < now - 10.0:
            self.nod_times.popleft()
        nod_recent = len(self.nod_times)

        looking_away = abs(yaw) >= self.yaw_lookaway_th
        if looking_away:
            if self._lookaway_t0 is None:
                self._lookaway_t0 = now
        else:
            self._lookaway_t0 = None
        lookaway_dur = (now - self._lookaway_t0) if self._lookaway_t0 else 0.0

        alert, text = self._decide(
            microsleep_s, perclos, yawn_3min, yawn_s, nod_recent, svm_drowsy, lookaway_dur
        )
        if alert != AlertLevel.NONE and self._last_alert == AlertLevel.NONE:
            self._alert_since = now
        if alert == AlertLevel.NONE:
            self._alert_since = 0.0
            self._last_beep = 0.0
        self._last_alert = alert

        return FusionState(
            blinks=self.blinks,
            yawns=self.yawns,
            microsleep_s=microsleep_s,
            yawn_duration_s=yawn_s,
            perclos_60=perclos,
            blink_rate=float(blink_rate),
            nod_count=self.nods,
            eyes_closed=self.eyes_closed,
            yawning=self.yawning,
            alert=alert,
            alert_text=text,
            svm_drowsy=svm_drowsy,
        )

    def _decide(
        self,
        microsleep_s: float,
        perclos: float,
        yawn_3min: int,
        yawn_s: float,
        nod_recent: int,
        svm_drowsy: bool | None,
        lookaway_dur: float = 0.0,
    ) -> tuple[AlertLevel, str]:
        if microsleep_s >= self.microsleep_th:
            return AlertLevel.STRONG, f"ALERT MANH: microsleep >= {self.microsleep_th:.1f}s (mat nham keo dai)"
        if perclos > self.perclos_th:
            return AlertLevel.FATIGUE, "ALERT MET: PERCLOS 60s > 30%"
        if yawn_3min >= 3 or yawn_s >= self.yawn_long_s:
            return AlertLevel.EARLY, "CANH BAO SOM: ngap/phut (3/3ph) hoac MAR >= 3s"
        if nod_recent >= 2:
            return AlertLevel.EARLY, "CANH BAO: gat dau lap lai"
        if lookaway_dur >= self.lookaway_s:
            return AlertLevel.EARLY, "CANH BAO: nhin lech duong (yaw dau)"
        if svm_drowsy:
            return AlertLevel.EARLY, "CANH BAO SOM: SVM/RF du doan buon ngu"
        return AlertLevel.NONE, ""

    def should_beep(self, now: float, min_interval: float | None = None) -> bool:
        if self._last_alert == AlertLevel.NONE:
            return False
        if min_interval is None:
            # Tần số còi phụ thuộc vào độ nguy hiểm:
            # - STRONG (microsleep/mất lái): 0.8s dồn dập
            # - FATIGUE (mệt mỏi/PERCLOS cao): 1.5s
            # - EARLY (ngáp/quay mặt): 2.5s
            intervals = {
                AlertLevel.STRONG: 0.8,
                AlertLevel.FATIGUE: 1.5,
                AlertLevel.EARLY: 2.5,
            }
            min_interval = intervals.get(self._last_alert, 2.0)
        if self._last_beep <= 0 or (now - self._last_beep) >= min_interval:
            self._last_beep = now
            return True
        return False
