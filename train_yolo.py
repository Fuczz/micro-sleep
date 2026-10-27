"""Train YOLO mắt / ngáp — khuyến nghị 50–100 epoch, yaml tách riêng."""

from __future__ import annotations

import argparse

from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["eye", "yawn"], required=True)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--model", default="yolov8n.pt")
    p.add_argument("--imgsz", type=int, default=320)
    args = p.parse_args()

    data = "dataset_eye.yaml" if args.task == "eye" else "dataset_yawn.yaml"
    project = "runs/detecteye" if args.task == "eye" else "runs/detectyawn"
    print(f"Train {args.task}: data={data} epochs={args.epochs}")
    model = YOLO(args.model)
    model.train(
        data=data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        project=project,
        name="train",
        exist_ok=True,
        plots=True,
    )
    model.val(data=data, split="test", plots=True)
    print("Xong. Kiểm tra confusion matrix trong thư mục runs/...")


if __name__ == "__main__":
    main()
