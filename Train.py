"""
train.py
Fine-tunes an ImageNet-pretrained ResNet-50 on a small (10-class, ~100 img/class)
fine-grained species dataset.

Expected data layout (ImageFolder style):
    data/
        train/
            class_a/ img1.jpg img2.jpg ...
            class_b/ ...
            ...

Strategy for small fine-grained data:
    1. Stratified train/val split (per-class) since dataset is tiny.
    2. Structure-preserving augmentation (mild crop/flip/color jitter -
       fine-grained cues like texture/color are easy to destroy with heavy aug).
    3. Two-phase progressive fine-tuning:
         Phase 1: freeze backbone, train only the new FC head (fast warmup).
         Phase 2: unfreeze layer3+layer4 (+fc), fine-tune with discriminative
                   learning rates (backbone << head) since low-level ImageNet
                   features transfer fine, only high-level features need updating.
    4. Label smoothing + weight decay + cosine LR schedule to fight overfitting.
    5. Early stopping on val accuracy, best checkpoint saved to outputs/model.pt.

Run:
    python train.py --data_dir data/train --output_dir outputs
"""

import argparse
import copy
import os
import random
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, models

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_transforms(img_size: int):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0), ratio=(0.85, 1.15)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.08)),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, val_tf


class TransformSubset(Dataset):
    """Wrap a Subset of samples (path, label) with its own transform."""

    def __init__(self, samples, loader, transform):
        self.samples = samples
        self.loader = loader
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = self.loader(path)
        img = self.transform(img)
        return img, label


def stratified_split(dataset: datasets.ImageFolder, val_split: float, seed: int):
    by_class = defaultdict(list)
    for path, label in dataset.samples:
        by_class[label].append((path, label))

    rng = random.Random(seed)
    train_samples, val_samples = [], []
    for label, items in by_class.items():
        rng.shuffle(items)
        n_val = max(1, int(round(len(items) * val_split)))
        val_samples.extend(items[:n_val])
        train_samples.extend(items[n_val:])
    return train_samples, val_samples


def build_model(num_classes: int, dropout: float = 0.3):
    weights = models.ResNet50_Weights.IMAGENET1K_V2
    model = models.resnet50(weights=weights)

    # freeze everything by default; unfreeze selectively later
    for p in model.parameters():
        p.requires_grad = False

    in_feats = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_feats, num_classes),
    )
    return model


def set_phase(model: nn.Module, phase: int):
    """phase 1: only fc trainable. phase 2: layer3, layer4, fc trainable."""
    for p in model.parameters():
        p.requires_grad = False
    for p in model.fc.parameters():
        p.requires_grad = True
    if phase == 2:
        for p in model.layer4.parameters():
            p.requires_grad = True
        for p in model.layer3.parameters():
            p.requires_grad = True


def make_optimizer(model, phase: int, head_lr: float, backbone_lr: float, weight_decay: float):
    if phase == 1:
        params = [{"params": model.fc.parameters(), "lr": head_lr}]
    else:
        params = [
            {"params": model.fc.parameters(), "lr": head_lr},
            {"params": model.layer4.parameters(), "lr": backbone_lr},
            {"params": model.layer3.parameters(), "lr": backbone_lr * 0.5},
        ]
    return torch.optim.AdamW(params, weight_decay=weight_decay)


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train(mode=train)
    total_loss, total_correct, total_n = 0.0, 0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            if train:
                loss.backward()
                optimizer.step()

        preds = outputs.argmax(dim=1)
        total_loss += loss.item() * imgs.size(0)
        total_correct += (preds == labels).sum().item()
        total_n += imgs.size(0)

    return total_loss / total_n, total_correct / total_n


def train_phase(model, phase, epochs, train_loader, val_loader, criterion, device,
                 head_lr, backbone_lr, weight_decay, patience, best_state, best_acc):
    set_phase(model, phase)
    optimizer = make_optimizer(model, phase, head_lr, backbone_lr, weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    epochs_no_improve = 0
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step()

        print(f"[phase {phase}] epoch {epoch}/{epochs} "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
              f"({time.time() - t0:.1f}s)")

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"[phase {phase}] early stopping (no improvement for {patience} epochs)")
                break

    return best_state, best_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/train")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--phase1_epochs", type=int, default=8)
    parser.add_argument("--phase2_epochs", type=int, default=20)
    parser.add_argument("--head_lr", type=float, default=1e-3)
    parser.add_argument("--backbone_lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    base_dataset = datasets.ImageFolder(args.data_dir)
    class_to_idx = base_dataset.class_to_idx
    num_classes = len(class_to_idx)
    print(f"found {len(base_dataset.samples)} images across {num_classes} classes")

    train_samples, val_samples = stratified_split(base_dataset, args.val_split, args.seed)
    train_tf, val_tf = get_transforms(args.img_size)

    train_ds = TransformSubset(train_samples, base_dataset.loader, train_tf)
    val_ds = TransformSubset(val_samples, base_dataset.loader, val_tf)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    print(f"train samples: {len(train_ds)}, val samples: {len(val_ds)}")

    model = build_model(num_classes, dropout=args.dropout).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    best_state = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # Phase 1: train head only
    best_state, best_acc = train_phase(
        model, 1, args.phase1_epochs, train_loader, val_loader, criterion, device,
        args.head_lr, args.backbone_lr, args.weight_decay, args.patience, best_state, best_acc,
    )

    # Phase 2: unfreeze layer3/layer4, discriminative LR fine-tuning
    model.load_state_dict(best_state)
    best_state, best_acc = train_phase(
        model, 2, args.phase2_epochs, train_loader, val_loader, criterion, device,
        args.head_lr, args.backbone_lr, args.weight_decay, args.patience, best_state, best_acc,
    )

    ckpt_path = os.path.join(args.output_dir, "model.pt")
    torch.save({
        "model_state_dict": best_state,
        "class_to_idx": class_to_idx,
        "img_size": args.img_size,
        "arch": "resnet50",
        "num_classes": num_classes,
        "best_val_acc": best_acc,
    }, ckpt_path)

    print(f"best val_acc: {best_acc:.4f}")
    print(f"saved model to {ckpt_path}")


if __name__ == "__main__":
    main()
