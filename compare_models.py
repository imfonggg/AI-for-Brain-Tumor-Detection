"""
Put the test results of several trained models side by side.

Usage (after running evaluate.py with --output_csv for each model):
    python compare_models.py outputs/resnet50_metrics.csv outputs/efficientnet_b0_metrics.csv

Prints an overall table and a per-class table, and saves
outputs/model_comparison.csv and outputs/model_comparison.png.
"""

import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = "outputs"
METRICS = ["precision", "recall", "f1"]


def read_metrics(path):
    with open(path, newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_files", nargs="+",
                        help="Metrics CSVs written by evaluate.py --output_csv")
    args = parser.parse_args()

    rows = [row for path in args.csv_files for row in read_metrics(path)]
    overall = [row for row in rows if row["scope"] == "overall"]
    per_class = [row for row in rows if row["scope"] == "class"]
    models = [row["model"] for row in overall]
    classes = sorted({row["class"] for row in per_class})

    print("\nOverall (Testing/ set):")
    print(f"  {'model':<18}{'accuracy':>10}{'precision':>11}{'recall':>9}{'f1':>9}")
    for row in overall:
        print(f"  {row['model']:<18}{float(row['accuracy']):>10.4f}"
              f"{float(row['precision']):>11.4f}{float(row['recall']):>9.4f}"
              f"{float(row['f1']):>9.4f}")

    print("\nPer-class recall (how many of each class were caught):")
    print(f"  {'class':<14}" + "".join(f"{m:>18}" for m in models))
    for cls in classes:
        values = {row["model"]: float(row["recall"]) for row in per_class
                  if row["class"] == cls}
        print(f"  {cls:<14}" + "".join(f"{values.get(m, float('nan')):>18.4f}"
                                       for m in models))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_csv = os.path.join(OUTPUT_DIR, "model_comparison.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCombined metrics saved to {out_csv}")

    # One panel per metric, grouped bars per class, one bar per model
    fig, axes = plt.subplots(1, len(METRICS), figsize=(5 * len(METRICS), 4), sharey=True)
    x = np.arange(len(classes))
    width = 0.8 / len(models)
    for ax, metric in zip(axes, METRICS):
        for i, model in enumerate(models):
            values = [next((float(r[metric]) for r in per_class
                            if r["model"] == model and r["class"] == cls), 0.0)
                      for cls in classes]
            ax.bar(x + (i - (len(models) - 1) / 2) * width, values, width, label=model)
        ax.set_xticks(x)
        ax.set_xticklabels(classes)
        ax.set_title(metric.capitalize())
        ax.set_ylim(0, 1)
    axes[0].legend()
    fig.suptitle("Per-class test metrics by model")
    fig.tight_layout()
    out_png = os.path.join(OUTPUT_DIR, "model_comparison.png")
    fig.savefig(out_png, dpi=150)
    print(f"Comparison chart saved to {out_png}")


if __name__ == "__main__":
    main()
