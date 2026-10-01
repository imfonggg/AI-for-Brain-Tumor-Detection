"""
Evaluation script -- run this on the held-out Testing/ set after training.

Usage:
    python evaluate.py --checkpoint outputs/resnet50_best.pt
"""

import argparse
import os

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, precision_recall_fscore_support)

from data_loader import get_dataloaders
from model import build_resnet50

OUTPUT_DIR = "outputs"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, test_loader, class_names = get_dataloaders(
        data_dir=args.data_dir, batch_size=args.batch_size
    )

    checkpoint = torch.load(args.checkpoint, map_location=device)
    class_names = checkpoint.get("class_names", class_names)

    model = build_resnet50(num_classes=len(class_names), pretrained=False).to(device)
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

    print("\n=== ResNet50 results on Testing/ set ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision (weighted): {precision:.4f}")
    print(f"Recall (weighted):    {recall:.4f}")
    print(f"F1 (weighted):        {f1:.4f}")
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
    plt.title("Confusion Matrix: ResNet50")
    plt.tight_layout()
    cm_path = os.path.join(OUTPUT_DIR, "resnet50_confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    print(f"\nConfusion matrix saved to {cm_path}")


if __name__ == "__main__":
    main()
