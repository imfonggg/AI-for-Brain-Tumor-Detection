"""Writes one evaluation results CSV per run (pure Python, no torch needed)."""

import csv
import os
import re

COLUMNS = [
    "class",
    "train_count", "train_percent", "val_count", "val_percent",
    "test_precision", "test_recall", "test_f1", "test_support",
    "best_val_accuracy", "test_accuracy",
]


def count_labels(dataset, num_classes):
    """Count images per class in an ImageFolder or a Subset of one."""
    if hasattr(dataset, "indices"):  # Subset from random_split
        labels = [dataset.dataset.targets[i] for i in dataset.indices]
    else:
        labels = dataset.targets
    counts = [0] * num_classes
    for label in labels:
        counts[label] += 1
    return counts


def results_path(output_dir, model, run):
    return os.path.join(output_dir, f"{model}_{run}_results.csv")


def next_run(output_dir, model):
    """run1, run2, ... : one past the highest runN results file for this model."""
    pattern = re.compile(rf"^{re.escape(model)}_run(\d+)_results\.csv$")
    highest = 0
    if os.path.isdir(output_dir):
        for name in os.listdir(output_dir):
            match = pattern.match(name)
            if match:
                highest = max(highest, int(match.group(1)))
    return f"run{highest + 1}"


def _pct(count, total):
    return round(100 * count / total, 1) if total else ""


def write_results(path, class_names, train_counts, val_counts, report,
                  test_accuracy, best_val_accuracy):
    """Write the results table for one run of one model.

    One row per class, then macro avg, weighted avg, and overall.
    report: sklearn classification_report(..., output_dict=True)
    Refuses to overwrite an existing file.
    """
    if os.path.exists(path):
        raise FileExistsError(f"{path} already exists. Use a different --run_label.")

    n_train, n_val = sum(train_counts), sum(val_counts)
    rows = []
    for i, name in enumerate(class_names):
        r = report[name]
        rows.append([name,
                     train_counts[i], _pct(train_counts[i], n_train),
                     val_counts[i], _pct(val_counts[i], n_val),
                     round(r["precision"], 4), round(r["recall"], 4),
                     round(r["f1-score"], 4), int(r["support"]), "", ""])
    for label in ("macro avg", "weighted avg"):
        r = report[label]
        rows.append([label, "", "", "", "",
                     round(r["precision"], 4), round(r["recall"], 4),
                     round(r["f1-score"], 4), int(r["support"]), "", ""])
    rows.append(["overall", n_train, "", n_val, "", "", "", "",
                 int(report["macro avg"]["support"]),
                 round(float(best_val_accuracy), 4), round(float(test_accuracy), 4)])

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(COLUMNS)
        writer.writerows(rows)