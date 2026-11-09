"""
Leave-one-source-out (LOSO) evaluation.

For each source dataset S in manifest.csv:
    train on every slice that is NOT from S (Training/ split, val split by patient),
    test on every slice from S (Training/ and Testing/: none of it was seen).

The usual test set shares sources with the training set, so a model that
recognises the source still scores well on it. Here the test source is new, so
the source fingerprint is useless and accuracy usually drops. That lower number
is the honest estimate of how the model does on a new hospital's scans.

A class can only be tested on S if some OTHER source also has that class;
otherwise the model never saw it during training. Those classes are skipped and
listed ("uncovered"), and a source with no covered class is skipped entirely.
This is also a quick way to see which classes still need a second source.

Usage:
    python loso.py --data_dir data_real --model resnet50 --head_epochs 3 --epochs 5
    python loso.py --sources figshare-cheng brats-men      # only these hold-outs
    python loso.py --dry_run                               # just show the plan

Outputs: outputs/loso_<model>.csv (one row per held-out source and class) and
outputs/loso_<model>_summary.csv. Needs a GPU in practice: one full training run
per held-out source.
"""

import argparse
import csv
import os
from collections import defaultdict

from data_loader import load_manifest, loaders_from_samples

OUTPUT_DIR = "outputs"


def plan(rows, sources=None, min_slices=50):
    """[(source, covered_classes, uncovered_classes)] for each hold-out."""
    classes_by_source = defaultdict(set)
    count = defaultdict(int)
    for r in rows:
        classes_by_source[r["source"]].add(r["class"])
        count[r["source"]] += 1
    out = []
    for s in sorted(classes_by_source):
        if sources and s not in sources:
            continue
        if count[s] < min_slices:
            continue
        others = set().union(*(c for t, c in classes_by_source.items() if t != s))
        covered = sorted(classes_by_source[s] & others)
        uncovered = sorted(classes_by_source[s] - others)
        out.append((s, covered, uncovered))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_dir", default="data_real")
    parser.add_argument("--model", default="resnet50")
    parser.add_argument("--head_epochs", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--sources", nargs="*", default=None,
                        help="Only hold out these sources (default: all)")
    parser.add_argument("--min_slices", type=int, default=50,
                        help="Skip sources with fewer slices than this")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print what would be trained and tested, then stop")
    args = parser.parse_args()

    rows = load_manifest(args.data_dir)
    if not rows:
        raise SystemExit(f"No manifest.csv in {args.data_dir}. Rebuild the dataset with the "
                         f"current nifti_slices.py, which records each slice's source.")
    class_names = sorted({r["class"] for r in rows})
    label = {c: i for i, c in enumerate(class_names)}
    print(f"classes: {class_names}")

    runs = plan(rows, args.sources, args.min_slices)
    print("\nLOSO plan:")
    for s, covered, uncovered in runs:
        note = f"   (skipped classes, no other source: {uncovered})" if uncovered else ""
        print(f"  hold out {s:<16} test classes {covered or 'none -> skip'}{note}")
    if args.dry_run:
        return

    import torch
    from train import train_model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    detail_path = os.path.join(OUTPUT_DIR, f"loso_{args.model}.csv")
    summary_path = os.path.join(OUTPUT_DIR, f"loso_{args.model}_summary.csv")
    detail, summary = [], []

    for held_out, covered, uncovered in runs:
        if not covered:
            continue
        print(f"\n===== hold out {held_out} =====")
        train = [(r["abspath"], label[r["class"]]) for r in rows
                 if r["source"] != held_out and r["split"] == "Training"]
        test = [(r["abspath"], label[r["class"]]) for r in rows
                if r["source"] == held_out and r["class"] in covered]
        print(f"train slices {len(train)}, test slices {len(test)}")
        train_loader, val_loader, test_loader = loaders_from_samples(
            train, test, class_names, batch_size=args.batch_size)
        ckpt = os.path.join(OUTPUT_DIR, f"loso_{args.model}_{held_out}.pt")
        train_model(args.model, train_loader, val_loader, class_names, args.head_epochs,
                    args.epochs, device, ckpt, extra_checkpoint={"held_out_source": held_out})

        from model import build_model
        state = torch.load(ckpt, map_location=device)
        model = build_model(args.model, num_classes=len(class_names), pretrained=False)
        model.load_state_dict(state["model_state_dict"])
        model = model.to(device).eval()
        preds, labels = [], []
        with torch.no_grad():
            for images, y in test_loader:
                preds += model(images.to(device)).argmax(1).cpu().tolist()
                labels += y.tolist()
        acc = sum(p == y for p, y in zip(preds, labels)) / len(labels)
        recalls = {}
        for cls in covered:
            idx = [i for i, y in enumerate(labels) if y == label[cls]]
            recalls[cls] = sum(preds[i] == labels[i] for i in idx) / len(idx)
        bal = sum(recalls.values()) / len(recalls)   # mean recall over the tested classes
        print(f"{held_out}: accuracy {acc:.1%}, balanced accuracy {bal:.1%} on an unseen source")
        for cls, rec in recalls.items():
            idx = [i for i, y in enumerate(labels) if y == label[cls]]
            print(f"  {cls:<12} recall {rec:.1%} (n={len(idx)})")
            detail.append({"model": args.model, "held_out_source": held_out, "class": cls,
                           "recall": rec, "n": len(idx)})
        summary.append({"model": args.model, "held_out_source": held_out, "accuracy": acc,
                        "balanced_accuracy": bal, "n": len(labels),
                        "tested_classes": " ".join(covered),
                        "skipped_classes": " ".join(uncovered)})
        # write after every source so a Kaggle timeout keeps finished results
        for path, data in ((detail_path, detail), (summary_path, summary)):
            with open(path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(data[0]))
                w.writeheader()
                w.writerows(data)

    if summary:
        print("\nLOSO summary (compare with the normal test accuracy from evaluate.py):")
        for s in summary:
            print(f"  {s['held_out_source']:<16} acc {s['accuracy']:.1%}  "
                  f"balanced {s['balanced_accuracy']:.1%}  n={s['n']}")
        print(f"saved {detail_path} and {summary_path}")


if __name__ == "__main__":
    main()
