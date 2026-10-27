"""Lớp B — MobileNetV3 / EfficientNet-B0 trên ROI mắt hoặc miệng (crop từ label YOLO)."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from cv_core.dl_models import build_efficientnet, build_mobilenet


def yolo_crop(img_path: Path, label_path: Path) -> list[tuple[np.ndarray, int]]:
    img = cv2.imread(str(img_path))
    if img is None or not label_path.exists():
        return []
    h, w = img.shape[:2]
    out = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls, xc, yc, bw, bh = int(parts[0]), *map(float, parts[1:5])
        x1 = int((xc - bw / 2) * w)
        y1 = int((yc - bh / 2) * h)
        x2 = int((xc + bw / 2) * w)
        y2 = int((yc + bh / 2) * h)
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, w), min(y2, h)
        if x2 - x1 < 8 or y2 - y1 < 8:
            continue
        out.append((img[y1:y2, x1:x2], cls))
    return out


class RoiFolder(Dataset):
    def __init__(self, root: Path, split: str, tf):
        self.items: list[tuple[np.ndarray, int]] = []
        img_dir = root / "images" / split
        lab_dir = root / "labels" / split
        if not img_dir.exists():
            raise FileNotFoundError(f"Không thấy {img_dir}")
        for img_path in sorted(img_dir.glob("*")):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            lab = lab_dir / f"{img_path.stem}.txt"
            self.items.extend(yolo_crop(img_path, lab))
        self.tf = tf
        print(f"{split}: {len(self.items)} ROI")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        bgr, y = self.items[i]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return self.tf(rgb), y


def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            pred = model(x).argmax(1).cpu().numpy()
            ys.extend(y.numpy().tolist())
            ps.extend(pred.tolist())
    print(confusion_matrix(ys, ps))
    print(classification_report(ys, ps, digits=4))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["eye", "mouth"], default="eye")
    p.add_argument("--arch", choices=["mobilenet", "efficientnet"], default="mobilenet")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    args = p.parse_args()

    root = Path("datasets/eye" if args.task == "eye" else "datasets/yawn")
    out_dir = Path("runs/classify_eye" if args.task == "eye" else "runs/classify_mouth")
    out_dir.mkdir(parents=True, exist_ok=True)
    name = "mobilenet" if args.arch == "mobilenet" else "efficientnet"
    out_path = out_dir / f"{name}_{'eye' if args.task == 'eye' else 'mouth'}.pt"

    tf_train = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    tf_val = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    train_ds = RoiFolder(root, "train", tf_train)
    val_split = "val" if (root / "images" / "val").exists() else "train"
    val_ds = RoiFolder(root, val_split, tf_val)
    train_ld = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0)
    val_ld = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = (build_mobilenet(2) if args.arch == "mobilenet" else build_efficientnet(2)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        n = 0
        for x, y in train_ld:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            total += float(loss.item()) * y.size(0)
            n += y.size(0)
        print(f"epoch {epoch}/{args.epochs} loss={total / max(n, 1):.4f}")

    torch.save(model.state_dict(), out_path)
    print("Saved", out_path)
    print("Confusion matrix (val):")
    evaluate(model, val_ld, device)
    test_dir = root / "images" / "test"
    if test_dir.exists():
        print("Confusion matrix (test):")
        test_ds = RoiFolder(root, "test", tf_val)
        evaluate(model, DataLoader(test_ds, batch_size=args.batch), device)


if __name__ == "__main__":
    main()
