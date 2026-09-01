"""Fine-tune a ResNet18 classifier on the real/fake face subset.

Intended to run on Colab (GPU); falls back to MPS/CPU locally via get_device().
Saves only the best-val-accuracy checkpoint to keep the repo small.
"""
import argparse
import time

import torch
import torch.nn as nn
from torch.optim import Adam

from src.common.paths import MODELS_DIR
from src.preprocessing.dataset import build_dataloaders
from src.training.model import build_model, get_device


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(train):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            if train:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--freeze-backbone", action="store_true")
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    loaders, classes = build_dataloaders(batch_size=args.batch_size)
    print(f"Classes: {classes}")

    model = build_model(freeze_backbone=args.freeze_backbone).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    best_val_acc = 0.0
    checkpoint_path = MODELS_DIR / "best_model.pth"

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss, train_acc = run_epoch(model, loaders["train"], criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, loaders["valid"], criterion, optimizer, device, train=False)
        elapsed = time.time() - start

        print(
            f"epoch {epoch}/{args.epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} ({elapsed:.1f}s)"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"model_state": model.state_dict(), "classes": classes}, checkpoint_path)
            print(f"  saved new best checkpoint (val_acc={val_acc:.4f})")

    print(f"Best val_acc: {best_val_acc:.4f}. Checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
