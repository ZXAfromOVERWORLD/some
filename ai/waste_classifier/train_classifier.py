#!/usr/bin/env python3
"""
Train a learned waste image classifier from a folder dataset.

Expected dataset formats:
1) root/train/<class_name>/*.jpg and root/val/<class_name>/*.jpg
2) root/<class_name>/*.jpg  (script will split into train/val)
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms


def _pick_existing_dir(base: Path, names: list[str]) -> Path | None:
    for n in names:
        p = base / n
        if p.is_dir():
            return p
    return None


def _make_loaders(dataset_dir: Path, batch_size: int, val_split: float):
    # Supports:
    # 1) dataset root (with train/val or TRAIN/TEST)
    # 2) direct train path (e.g. .../DATASET/train or .../DATASET/TRAIN)
    lower_name = dataset_dir.name.lower()
    if lower_name == "train":
        train_dir = dataset_dir
        parent = dataset_dir.parent
        val_dir = _pick_existing_dir(parent, ["val", "VAL", "test", "TEST"])
    else:
        train_dir = _pick_existing_dir(dataset_dir, ["train", "TRAIN"])
        val_dir = _pick_existing_dir(dataset_dir, ["val", "VAL", "test", "TEST"])

    tf_train = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(12),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    tf_eval = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    if train_dir is not None and train_dir.is_dir() and val_dir is not None and val_dir.is_dir():
        ds_train = datasets.ImageFolder(str(train_dir), transform=tf_train)
        ds_val = datasets.ImageFolder(str(val_dir), transform=tf_eval)
    else:
        # If `dataset_dir` points directly to train folder, split that folder.
        split_source = train_dir if (train_dir is not None and train_dir.is_dir()) else dataset_dir
        ds_all = datasets.ImageFolder(str(split_source), transform=tf_train)
        n_total = len(ds_all)
        n_val = max(1, int(n_total * val_split))
        n_train = max(1, n_total - n_val)
        ds_train, ds_val = random_split(ds_all, [n_train, n_val], generator=torch.Generator().manual_seed(42))
        # set eval transform for val subset
        ds_val.dataset.transform = tf_eval

    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=2)
    class_names = ds_train.dataset.classes if hasattr(ds_train, "dataset") else ds_train.classes
    return train_loader, val_loader, class_names


def _eval(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            correct += int((pred == y).sum().item())
            total += int(y.numel())
    return (correct / total) if total else 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, help="Dataset root path")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-split", type=float, default=0.2)
    p.add_argument("--out", default="models/waste_cls_best.pt")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_dir = Path(args.dataset)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_dir}")

    train_loader, val_loader, class_names = _make_loaders(dataset_dir, args.batch_size, args.val_split)

    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(in_features, len(class_names))
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    best_val = -1.0
    os.makedirs(Path(args.out).parent, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            bsz = int(y.numel())
            running_loss += float(loss.item()) * bsz
            seen += bsz

        val_acc = _eval(model, val_loader, device)
        train_loss = (running_loss / seen) if seen else 0.0
        print(f"epoch={epoch} train_loss={train_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val:
            best_val = val_acc
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "class_names": class_names,
                    "val_acc": best_val,
                },
                args.out,
            )
            print(f"saved_best={args.out} val_acc={best_val:.4f}")

    print(f"done best_val_acc={best_val:.4f}")


if __name__ == "__main__":
    main()

