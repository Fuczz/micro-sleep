"""Lớp C — SVM / Random Forest trên [EAR, MAR, pitch, blink_rate, PERCLOS].

Cần CSV có cột label (0=tỉnh, 1=buồn ngủ). Có thể gán tay trên logs/session_*.csv
hoặc thêm cột khi thu thập.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

FEATURES = ["ear", "mar", "pitch", "blink_rate", "perclos_60"]
ALT = {"perclos": "perclos_60", "perclos60": "perclos_60"}


def load_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    for src, dst in ALT.items():
        if src in df.columns and dst not in df.columns:
            df[dst] = df[src]
    if "label" not in df.columns and "drowsy" in df.columns:
        df["label"] = df["drowsy"]
    missing = [c for c in FEATURES + ["label"] if c not in df.columns]
    if missing:
        raise ValueError(f"{path} thiếu cột {missing}. Cần {FEATURES} và label.")
    df = df.dropna(subset=FEATURES + ["label"])
    df["label"] = df["label"].astype(int)
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", nargs="*", default=[], help="File CSV đã gán nhãn")
    p.add_argument("--logs-dir", default="logs")
    args = p.parse_args()

    files = [Path(x) for x in args.csv]
    if not files:
        files = list(Path(args.logs_dir).glob("*.csv"))
    labeled = []
    for f in files:
        try:
            df = load_table(f)
            labeled.append(df)
            print(f"OK {f}: {len(df)} dòng")
        except Exception as exc:
            print(f"Bỏ {f}: {exc}")

    if not labeled:
        print(
            "Chưa có CSV có cột label.\n"
            "Cách làm: chạy DrowsinessDetector.py → logs/session_*.csv\n"
            "Thêm cột label (0 tỉnh / 1 buồn ngủ) rồi:\n"
            "  python train_traditional.py --csv logs/session_xxx.csv"
        )
        return

    data = pd.concat(labeled, ignore_index=True)
    x = data[FEATURES].to_numpy(dtype=np.float32)
    y = data["label"].to_numpy(dtype=np.int64)
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.25, random_state=42, stratify=y)

    out = Path("runs/traditional")
    out.mkdir(parents=True, exist_ok=True)

    svm = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=2.0, probability=True, class_weight="balanced")),
        ]
    )
    rf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")

    for name, model, path in (
        ("SVM", svm, out / "drowsiness_svm.joblib"),
        ("RandomForest", rf, out / "drowsiness_rf.joblib"),
    ):
        model.fit(x_tr, y_tr)
        pred = model.predict(x_te)
        print(f"\n=== {name} ===")
        print(confusion_matrix(y_te, pred))
        print(classification_report(y_te, pred, digits=4))
        joblib.dump(model, path)
        print("Saved", path)


if __name__ == "__main__":
    main()
