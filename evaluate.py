"""
Evaluation script -- run this on the held-out Testing/ set after training.

Usage:
    python evaluate.py --checkpoint outputs/resnet50_best.pt
    python evaluate.py --checkpoint outputs/efficientnet_b0_best.pt

The architecture is read from the checkpoint's "model_name" field (saved by
train.py), so --model does not normally need to be passed -- it's only there
as a fallback for older checkpoints that predate that field.
"""

import argparse
import os

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, precision_recall_fscore_support)

from data_loader import get_dataloaders
from model import build_model, CLASSIFIER_ATTR

OUTPUT_DIR = "outputs"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--model", default=None, choices=sorted(CLASSIFIER_ATTR),
                         help="Override architecture if the checkpoint has no "
                              "'model_name' field (older checkpoints only)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, test_loader, class_names = get_dataloaders(
        data_dir=args.data_dir, batch_size=args.batch_size
    )

    checkpoint = torch.load(args.checkpoint, map_location=device)
    class_names = checkpoint.get("class_names", class_names)
    model_name = checkpoint.get("model_name", args.model)
    if model_name is None:
        raise ValueError(
            "Checkpoint has no 'model_name' field; pass --model explicitly "
            "(resnet50 or efficientnet_b0)."
        )

    model = build_model(model_name, num_classes=len(class_names), pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="weighted", zero_division=0
    )

    print(f"\n=== {model_name} results on Testing/ set ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision (weighted): {precision:.4f}")
    print(f"Recall (weighted): {recall:.4f}")
    print(f"F1 (weighted): {f1:.4f}")
    print("\nPer-class report:")
    print(classification_report(all_labels, all_preds, target_names=class_names,
                                 zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix: {model_name}")
    plt.tight_layout()
    cm_path = os.path.join(OUTPUT_DIR, f"{model_name}_confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    print(f"\nConfusion matrix saved to {cm_path}")


if __name__ == "__main__":
    main()