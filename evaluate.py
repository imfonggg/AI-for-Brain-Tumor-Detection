"""
Model Evaluation Suite & Clinical Metrics Module (Section 2.2 of Master Doc).

Calculates:
- Accuracy: Overall fraction of correct predictions.
- Precision (PPV): Reliability of positive tumor alerts (minimizes false alarms).
- Recall / Sensitivity (TPR): Proportion of actual tumor cases caught (safety metric).
- Specificity (TNR): Effectiveness in identifying healthy scans.
- F1-Score: Harmonic mean of precision and recall.
- Confusion Matrix: Error cross-tabulation matrix with visual Seaborn heatmap.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

import config


def compute_clinical_metrics(y_true: List[int], y_pred: List[int], class_names: List[str]) -> Dict[str, Any]:
    """
    Computes comprehensive clinical metrics including Specificity per class.
    """
    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)
    num_classes = len(class_names)

    # Global metrics
    accuracy = float(accuracy_score(y_true_np, y_pred_np))
    precision_macro = float(precision_score(y_true_np, y_pred_np, average="macro", zero_division=0))
    recall_macro = float(recall_score(y_true_np, y_pred_np, average="macro", zero_division=0))
    f1_macro = float(f1_score(y_true_np, y_pred_np, average="macro", zero_division=0))

    precision_weighted = float(precision_score(y_true_np, y_pred_np, average="weighted", zero_division=0))
    recall_weighted = float(recall_score(y_true_np, y_pred_np, average="weighted", zero_division=0))
    f1_weighted = float(f1_score(y_true_np, y_pred_np, average="weighted", zero_division=0))

    # Confusion matrix: rows = true classes, columns = predicted classes
    cm = confusion_matrix(y_true_np, y_pred_np, labels=list(range(num_classes)))

    # Compute per-class specificity and sensitivity (Recall)
    # For multi-class One-vs-Rest calculation:
    # TP = cm[i, i]
    # FP = sum(cm[:, i]) - TP
    # FN = sum(cm[i, :]) - TP
    # TN = total - (TP + FP + FN)
    total_samples = len(y_true_np)
    per_class_metrics = {}
    specificities = []

    for i, class_name in enumerate(class_names):
        tp = float(cm[i, i])
        fp = float(np.sum(cm[:, i]) - tp)
        fn = float(np.sum(cm[i, :]) - tp)
        tn = float(total_samples - (tp + fp + fn))

        sens = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        spec = (tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        prec = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        f1 = (2 * prec * sens / (prec + sens)) if (prec + sens) > 0 else 0.0

        specificities.append(spec)
        per_class_metrics[class_name] = {
            "TP": int(tp),
            "FP": int(fp),
            "FN": int(fn),
            "TN": int(tn),
            "Precision (PPV)": round(prec, 4),
            "Recall / Sensitivity (TPR)": round(sens, 4),
            "Specificity (TNR)": round(spec, 4),
            "F1-Score": round(f1, 4),
            "Support": int(tp + fn)
        }

    macro_specificity = float(np.mean(specificities))

    summary_metrics = {
        "Overall Accuracy": round(accuracy, 4),
        "Macro Precision": round(precision_macro, 4),
        "Macro Recall (Sensitivity)": round(recall_macro, 4),
        "Macro Specificity": round(macro_specificity, 4),
        "Macro F1-Score": round(f1_macro, 4),
        "Weighted Precision": round(precision_weighted, 4),
        "Weighted Recall": round(recall_weighted, 4),
        "Weighted F1-Score": round(f1_weighted, 4),
        "Per-Class Breakdown": per_class_metrics,
        "Confusion Matrix": cm.tolist()
    }

    return summary_metrics


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], save_path: Path) -> None:
    """
    Renders and saves a clinical confusion matrix heatmap showing counts and percentages.
    """
    plt.figure(figsize=(9, 7))
    
    # Normalized row-wise for display percentage
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)

    annot_matrix = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot_matrix[i, j] = f"{cm[i, j]}\n({cm_norm[i, j]*100:.1f}%)"

    sns.heatmap(
        cm_norm,
        annot=annot_matrix,
        fmt="",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True
    )
    plt.title("ResNet-50 Brain Tumor Detection - Test Confusion Matrix", fontsize=14, pad=15)
    plt.xlabel("Predicted Diagnosis", fontsize=12, labelpad=10)
    plt.ylabel("Ground Truth Diagnosis", fontsize=12, labelpad=10)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Metrics] Confusion matrix saved to: {save_path}")


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    class_names: List[str],
    device: torch.device = config.DEVICE
) -> Dict[str, Any]:
    """
    Runs model inference on the test set and calculates the full evaluation suite.
    """
    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []

    with torch.no_grad():
        for images, labels, _ in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())

    metrics = compute_clinical_metrics(y_true, y_pred, class_names)
    cm_array = np.array(metrics["Confusion Matrix"])

    plot_confusion_matrix(cm_array, class_names, config.CONFUSION_MATRIX_PATH)

    # Save metrics JSON for the team
    with open(config.METRICS_JSON_PATH, "w") as f:
        json.dump(metrics, f, indent=4)

    return metrics


def print_clinical_report(metrics: Dict[str, Any]) -> None:
    """
    Prints a formatted clinical summary directly to the terminal.
    """
    print("\n" + "=" * 65)
    print("      RESNET-50 BRAIN TUMOR CLINICAL EVALUATION SUITE")
    print("=" * 65)
    print(f"Overall Accuracy:            {metrics['Overall Accuracy'] * 100:.2f}%")
    print(f"Macro Precision (PPV):       {metrics['Macro Precision'] * 100:.2f}%  (Reliability of alerts)")
    print(f"Macro Recall / Sensitivity:  {metrics['Macro Recall (Sensitivity)'] * 100:.2f}%  (Core Safety: caught tumors)")
    print(f"Macro Specificity (TNR):     {metrics['Macro Specificity'] * 100:.2f}%  (Correctly identified healthy)")
    print(f"Macro F1-Score:              {metrics['Macro F1-Score'] * 100:.2f}%")
    print("-" * 65)
    print("PER-CLASS CLINICAL PERFORMANCE:")
    print(f"{'Class':<20} | {'Recall (TPR)':<12} | {'Precision':<10} | {'Specificity':<11} | {'F1':<6}")
    print("-" * 65)
    for cls_name, vals in metrics["Per-Class Breakdown"].items():
        rec = f"{vals['Recall / Sensitivity (TPR)'] * 100:.1f}%"
        prec = f"{vals['Precision (PPV)'] * 100:.1f}%"
        spec = f"{vals['Specificity (TNR)'] * 100:.1f}%"
        f1 = f"{vals['F1-Score'] * 100:.1f}%"
        print(f"{cls_name:<20} | {rec:<12} | {prec:<10} | {spec:<11} | {f1:<6}")
    print("=" * 65)
