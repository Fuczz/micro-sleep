"""Đánh giá đúng bài CV: confusion matrix, P/R/F1, FPS (nếu có log)."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="CSV có y_true và y_pred hoặc label + alert")
    p.add_argument("--y-true", default="label")
    p.add_argument("--y-pred", default="y_pred")
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    df.columns = [c.strip() for c in df.columns]
    if args.y_pred not in df.columns and "alert" in df.columns:
        df["y_pred"] = (df["alert"].astype(str) != "none").astype(int)
        args.y_pred = "y_pred"
    if args.y_true not in df.columns:
        raise SystemExit(f"Thiếu cột {args.y_true}")

    y_true = df[args.y_true].astype(int)
    y_pred = df[args.y_pred].astype(int)
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred, digits=4))
    if "fps" in df.columns:
        print(f"FPS trung bình: {pd.to_numeric(df['fps'], errors='coerce').mean():.2f}")
    if "dt_s" in df.columns:
        dt = pd.to_numeric(df["dt_s"], errors="coerce")
        print(f"Độ trễ frame trung bình: {dt.mean()*1000:.1f} ms")


if __name__ == "__main__":
    main()
