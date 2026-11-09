"""
Pass/fail checks for dataset shortcuts.

    python shortcut_checks.py shuffle --checkpoint outputs/resnet50_best.pt
    python shortcut_checks.py source

shuffle  Pixel-shuffle test. Shuffling the pixels of a slice destroys all anatomy
         and keeps only its brightness histogram. A model that reads the disease
         cannot tell the classes apart any more, so on shuffled test slices:
           * the true class should be predicted about 1/K of the time for every
             class (25% with 4 classes), and
           * balanced accuracy should be about chance.
         A class far above chance means the model recognises that class's source
         from its intensity "fingerprint". PASS needs every class within
         --tolerance of chance (default 0.15, so 10%-40% with 4 classes).

         The shuffled confusion matrix tells two failure modes apart:
           * one column full ("everything shuffled -> meningioma"): the model
             has a default answer for noise. Every class fails the per-class
             test, but balanced accuracy is at chance.
           * a strong diagonal: the model really reads the fingerprint.
             Balanced accuracy is above chance. This is the serious one.

         Two variants are run: "all pixels" (the original test) and "brain
         pixels" (only pixels inside the brain are shuffled, so the brain's
         outline stays). If only the brain-pixel variant fails, the histogram of
         brain tissue is the giveaway; if only the all-pixel one fails, the
         amount of black background is.

source   A random forest that sees only image statistics guesses the SOURCE
         dataset (from manifest.csv), and the CLASS from the same statistics.
         After harmonization both should drop. It also lists classes that still
         come from a single source; --strict fails while any remain.

Exit code is 1 when a check fails and --strict is given, so a notebook or CI
step can stop on it.
"""

import argparse
import csv
import glob
import json
import os
import random
import sys

import numpy as np
from PIL import Image

OUTPUT_DIR = "outputs"
INPUT_SIZE = 224


def load_gray(path):
    return np.asarray(Image.open(path).convert("L").resize((INPUT_SIZE, INPUT_SIZE)), np.uint8)


def shuffle_all(gray, rng):
    return rng.permutation(gray.ravel()).reshape(gray.shape)


def shuffle_brain(gray, rng, threshold=8):
    """Shuffle only the pixels inside the brain; background stays black."""
    out = gray.copy()
    inside = gray > threshold
    out[inside] = rng.permutation(gray[inside])
    return out


def sample_test_files(data_dir, per_class, seed):
    rng = random.Random(seed)
    files = {}
    for cls in sorted(os.listdir(os.path.join(data_dir, "Testing"))):
        paths = sorted(glob.glob(os.path.join(data_dir, "Testing", cls, "*.jpg")))
        files[cls] = rng.sample(paths, min(per_class, len(paths)))
    return files


def evaluate_shuffle(predict_fn, class_names, files, variant, repeats, seed, tolerance):
    """Returns (rows, confusion, balanced_acc, passed)."""
    k = len(class_names)
    chance = 1.0 / k
    rng = np.random.default_rng(seed)
    shuffle = shuffle_all if variant == "all pixels" else shuffle_brain
    confusion = np.zeros((k, k), int)
    rows = []
    for cls, paths in files.items():
        if cls not in class_names or not paths:
            continue
        target = class_names.index(cls)
        images = [shuffle(load_gray(p), rng) for p in paths for _ in range(repeats)]
        preds = predict_fn(images)
        for p in preds:
            confusion[target, p] += 1
        hit = float(np.mean(preds == target))
        top = int(np.bincount(preds, minlength=k).argmax())
        rows.append({"variant": variant, "class": cls, "n": len(images),
                     "true_class_rate": hit, "most_common_prediction": class_names[top],
                     "pass": abs(hit - chance) <= tolerance})
    recalls = [confusion[i, i] / confusion[i].sum() for i in range(k) if confusion[i].sum()]
    balanced = float(np.mean(recalls)) if recalls else float("nan")
    passed = all(r["pass"] for r in rows) and balanced <= chance + tolerance
    return rows, confusion, balanced, passed


def make_predictor(checkpoint, batch_size=64):
    import torch
    from data_loader import get_transforms
    from model import build_model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint, map_location=device)
    class_names = ckpt["class_names"]
    model = build_model(ckpt["model_name"], num_classes=len(class_names), pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()
    tf = get_transforms(train=False)

    @torch.no_grad()
    def predict(gray_images):
        out = []
        for i in range(0, len(gray_images), batch_size):
            batch = torch.stack([tf(Image.fromarray(g).convert("RGB"))
                                 for g in gray_images[i:i + batch_size]]).to(device)
            out.append(model(batch).argmax(1).cpu())
        return torch.cat(out).numpy()
    return predict, class_names, ckpt["model_name"]


def cmd_shuffle(args):
    predict, class_names, model_name = make_predictor(args.checkpoint)
    files = sample_test_files(args.data_dir, args.per_class, args.seed)
    k = len(class_names)
    print(f"Pixel-shuffle test: {model_name}, {k} classes, chance = {1 / k:.0%}, "
          f"pass band = {max(0, 1 / k - args.tolerance):.0%}-{1 / k + args.tolerance:.0%}\n")
    all_rows, summary, overall = [], {}, True
    for variant in ("all pixels", "brain pixels"):
        rows, confusion, balanced, passed = evaluate_shuffle(
            predict, class_names, files, variant, args.repeats, args.seed, args.tolerance)
        all_rows += rows
        overall &= passed
        summary[variant] = {"balanced_accuracy": balanced, "pass": passed,
                            "confusion": confusion.tolist()}
        print(f"--- shuffled {variant} ---")
        for r in rows:
            print(f"  {r['class']:<12} true class {r['true_class_rate']:6.1%}   "
                  f"most predicted: {r['most_common_prediction']:<12} "
                  f"{'PASS' if r['pass'] else 'FAIL'}")
        print(f"  balanced accuracy {balanced:.1%} (chance {1 / k:.0%})")
        print("  confusion (rows = true class, columns = predicted):", class_names)
        for name, row in zip(class_names, confusion):
            print(f"    {name:<12} {row.tolist()}")
        if not passed:
            if balanced > 1 / k + args.tolerance:
                print("  -> the model separates classes from brightness alone: it reads "
                      "the source fingerprint.")
            else:
                print("  -> balanced accuracy is near chance, so this is mostly a default "
                      "answer for noise, not a fingerprint. Still worth fixing (e.g. more "
                      "sources per class), but less serious.")
        print()
    print(f"OVERALL: {'PASS' if overall else 'FAIL'}")

    os.makedirs(args.output_dir, exist_ok=True)
    stem = os.path.join(args.output_dir, f"shuffle_test_{model_name}")
    with open(stem + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0]))
        w.writeheader()
        w.writerows(all_rows)
    with open(stem + ".json", "w") as f:
        json.dump({"model": model_name, "classes": class_names, "pass": overall,
                   "tolerance": args.tolerance, **summary}, f, indent=2)
    print(f"saved {stem}.csv and {stem}.json")
    return overall


# ---------- source test ----------
def image_features(path):
    """Statistics a dataset leaves behind regardless of tumor: brightness, contrast, black border, edges."""
    g = np.asarray(Image.open(path).convert("L").resize((128, 128)), dtype=np.float32)
    rows, cols = np.where(g > 10)
    box = ([rows.min(), rows.max(), cols.min(), cols.max()] if rows.size else [0, 0, 0, 0])
    ring = np.concatenate([g[:8].ravel(), g[-8:].ravel(), g[:, :8].ravel(), g[:, -8:].ravel()])
    hist = np.histogram(g, bins=32, range=(0, 255))[0] / g.size
    return np.concatenate([[g.mean(), g.std(), (g < 10).mean(), (g > 245).mean(), ring.mean(),
                            np.abs(np.diff(g, axis=1)).mean()],
                           np.percentile(g, [1, 5, 25, 50, 75, 95, 99]), np.array(box) / 128, hist])


def cmd_source(args):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import balanced_accuracy_score
    from data_loader import source_lookup
    source_of = source_lookup(args.data_dir)

    def load(split, seed):
        groups = {}
        for p in glob.glob(os.path.join(args.data_dir, split, "*", "*.jpg")):
            groups.setdefault((source_of(p), os.path.basename(os.path.dirname(p))), []).append(p)
        rng = random.Random(seed)
        chosen = [p for ps in groups.values() for p in rng.sample(ps, min(args.max_per_group, len(ps)))]
        X = np.stack([image_features(p) for p in chosen])
        return X, np.array([source_of(p) for p in chosen]), \
            np.array([os.path.basename(os.path.dirname(p)) for p in chosen])

    Xtr, src_tr, cls_tr = load("Training", 0)
    Xte, src_te, cls_te = load("Testing", 0)
    results = {}
    for target, ytr, yte in (("source", src_tr, src_te), ("class", cls_tr, cls_te)):
        clf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=0).fit(Xtr, ytr)
        bal = balanced_accuracy_score(yte, clf.predict(Xte))
        k = len(set(yte.tolist()))
        results[target] = {"balanced_accuracy": bal, "chance": 1 / k, "n_labels": k}
        print(f"{target} from image statistics alone: balanced accuracy {bal:.1%} "
              f"(chance {1 / k:.0%}, {k} labels)")
    print("\nSources per class:")
    for cls in sorted(set(cls_tr.tolist())):
        print(f"  {cls:<12} {sorted(set(src_tr[cls_tr == cls].tolist()))}")
    single = [c for c in set(cls_tr.tolist()) if len(set(src_tr[cls_tr == c].tolist())) == 1]
    if single:
        print(f"\nSingle-source classes: {sorted(single)}. Recognising the source is enough "
              f"to name these classes; add a second source for each.")
    # Informational, no pass/fail: tumors legitimately change the histogram a bit,
    # so the class guess will not be exactly at chance even with clean data.
    passed = not single
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "source_test.json"), "w") as f:
        json.dump({**results, "single_source_classes": sorted(single), "pass": passed}, f, indent=2)
    return passed


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("shuffle", help="pixel-shuffle pass/fail test for one checkpoint")
    s.add_argument("--checkpoint", required=True)
    s.add_argument("--per_class", type=int, default=200, help="test slices per class")
    s.add_argument("--repeats", type=int, default=1, help="shuffles per slice")
    s.add_argument("--tolerance", type=float, default=0.15)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_shuffle)
    r = sub.add_parser("source", help="can image statistics identify the source or class?")
    r.add_argument("--max_per_group", type=int, default=1500)
    r.set_defaults(func=cmd_source)
    for p in (s, r):
        p.add_argument("--data_dir", default="data_real")
        p.add_argument("--output_dir", default=OUTPUT_DIR)
        p.add_argument("--strict", action="store_true", help="exit with code 1 on FAIL")
    args = parser.parse_args()
    passed = args.func(args)
    if args.strict and not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
