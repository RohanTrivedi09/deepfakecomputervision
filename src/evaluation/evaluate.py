"""Evaluate the trained checkpoint on the held-out test split.

Writes reports/metrics.json (precision/recall/F1 per class + accuracy) and
reports/figures/confusion_matrix.png.
"""
import json

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import classification_report, confusion_matrix

from src.common.paths import FIGURES_DIR, MODELS_DIR, REPORTS_DIR
from src.preprocessing.dataset import build_dataloaders
from src.training.model import build_model, get_device


def main() -> None:
    device = get_device()
    loaders, classes = build_dataloaders()

    checkpoint = torch.load(MODELS_DIR / "best_model.pth", map_location=device)
    model = build_model().to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loaders["test"]:
            images = images.to(device)
            preds = model(images).argmax(1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    report = classification_report(all_labels, all_preds, target_names=classes, output_dict=True)
    cm = confusion_matrix(all_labels, all_preds)

    print(classification_report(all_labels, all_preds, target_names=classes))
    print("Confusion matrix:\n", cm)

    (REPORTS_DIR / "metrics.json").write_text(json.dumps(report, indent=2))

    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=150)
    print(f"Saved metrics.json and confusion_matrix.png to {REPORTS_DIR}")


if __name__ == "__main__":
    main()
