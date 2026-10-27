"""Xuất dataset mắt YOLO: CẢ open và closed, class khớp yaml.

Class (dataset_eye.yaml):
  0 = open eye
  1 = close eye

Lỗi cũ: chỉ ghi closed_eyes và gán class 0 (sai yaml) → model lệch.
"""

from __future__ import annotations

import argparse
import os

from datasets import DatasetDict, load_dataset


OPEN_LABELS = {"open", "open_eyes", "opened", "open eye", "awake", "0"}
CLOSED_LABELS = {"closed", "closed_eyes", "close", "close eye", "close_eyes", "drowsy", "1"}


def coco_to_yolo(x, y, w, h, width, height):
    return [((2 * x + w) / (2 * width)), ((2 * y + h) / (2 * height)), w / width, h / height]


def normalize_label(raw) -> str | None:
    name = str(raw).strip().lower().replace("-", "_")
    if name in OPEN_LABELS:
        return "open"
    if name in CLOSED_LABELS:
        return "closed"
    return None


def split_dataset(dataset, seed: int = 42):
    full = dataset["train"] if "train" in dataset else dataset
    split = full.train_test_split(test_size=0.2, seed=seed)
    train = split["train"]
    rest = split["test"].train_test_split(test_size=0.5, seed=seed)
    return DatasetDict({"train": train, "val": rest["train"], "test": rest["test"]})


def export_split(partition: str, data: DatasetDict, out_root: str, max_samples: int):
    img_dir = os.path.join(out_root, "images", partition)
    lab_dir = os.path.join(out_root, "labels", partition)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lab_dir, exist_ok=True)

    subset = data[partition]
    n = min(len(subset), max_samples)
    counts = {"open": 0, "closed": 0, "skip": 0}

    for i, sample in enumerate(subset.select(range(n))):
        kind = normalize_label(sample["Label"])
        if kind is None:
            counts["skip"] += 1
            continue
        class_id = 0 if kind == "open" else 1
        tag = "open" if kind == "open" else "close"

        img = sample["Image_data"]["file"]
        width, height = int(img.size[0]), int(img.size[1])
        boxes = [sample["Left_eye_react"], sample["Right_eye_react"]]

        lines = []
        for box in boxes:
            xc, yc, w, h = coco_to_yolo(box[0], box[1], box[2], box[3], width, height)
            lines.append(f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

        stem = f"{i:06d}_{tag}"
        with open(os.path.join(lab_dir, f"{stem}.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        img.save(os.path.join(img_dir, f"{stem}.png"))
        counts[kind] += 1

    print(f"[{partition}] open={counts['open']} closed={counts['closed']} skip={counts['skip']}")
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf", default="MichalMlodawski/closed-open-eyes")
    parser.add_argument("--out", default="datasets/eye")
    parser.add_argument("--max-samples", type=int, default=8000, help="Số mẫu tối đa mỗi split")
    args = parser.parse_args()

    print("Tải HuggingFace:", args.hf)
    raw = load_dataset(args.hf)
    split = split_dataset(raw)
    print("train/val/test:", len(split["train"]), len(split["val"]), len(split["test"]))

    for part in ("train", "val", "test"):
        export_split(part, split, args.out, args.max_samples)
    print("Xong. Dùng dataset_eye.yaml để train YOLO.")


if __name__ == "__main__":
    main()
