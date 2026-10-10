"""
Evaluation script -- run this on the held-out Testing/ set after training.

Usage:
    python evaluate.py --checkpoint outputs/resnet18_best.pt
    python evaluate.py --checkpoint outputs/efficientnet_b0_best.pt --run_label run2

Each run of each model gets its own file: outputs/{model}_{run}_results.csv
(7 rows: one per class, macro avg, weighted avg, overall). The run label is
auto-numbered (run1, run2, ... per model) unless you pass --run_label.
Existing files are never overwritten. Nothing is printed except the paths
written.

The architecture is read from the checkpoint's "model_name" field (saved by
train.py), so --model only matters for older checkpoints without that field.
"""

import argparse
import os

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from data_loader import get_dataloaders
from model import build_model, CLASSIFIER_ATTR
from results_csv import count_labels, next_run, results_path, write_results

OUTPUT_DIR = "outputs"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--run_label", default=None,
                         help="Name for this run (default: auto-numbered runN)")
    parser.add_argument("--model", default=None, choices=sorted(CLASSIFIER_ATTR),
                         help="Only needed for older checkpoints without 'model_name'")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        data_dir=args.data_dir, batch_size=args.batch_size, verbose=False
    )

    checkpoint = torch.load(args.checkpoint, map_location=device)
    class_names = checkpoint.get("class_names", class_names)
    model_name = checkpoint.get("model_name", args.model)
    if model_name is None:
        raise ValueError(
            "Checkpoint has no 'model_name' field; pass --model explicitly "
            "(resnet18, resnet50, or efficientnet_b0)."
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

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    report = classification_report(all_labels, all_preds, target_names=class_names,
                                    zero_division=0, output_dict=True)
    run = args.run_label or next_run(OUTPUT_DIR, model_name)
    csv_path = results_path(OUTPUT_DIR, model_name, run)

    write_results(
        csv_path, class_names,
        train_counts=count_labels(train_loader.dataset, len(class_names)),
        val_counts=count_labels(val_loader.dataset, len(class_names)),
        report=report,
        test_accuracy=accuracy_score(all_labels, all_preds),
        best_val_accuracy=checkpoint.get("val_acc", float("nan")),
    )

    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(class_names))))
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix: {model_name} ({run})")
    plt.tight_layout()
    png_path = os.path.join(OUTPUT_DIR, f"{model_name}_{run}_confusion_matrix.png")
    plt.savefig(png_path, dpi=150)

    print(f"{model_name} {run}: results saved to {csv_path}")
    print(f"Confusion matrix image saved to {png_path}")


if __name__ == "__main__":
    main()