#!/usr/bin/env python
"""NIfTI scans -> training slices (script version of the Kaggle notebook).

Builds the real-scan dataset used by train.py:

    data_real/Training/<class>/<patientID>_z###.jpg
    data_real/Testing/<class>/<patientID>_z###.jpg

Each patient goes entirely into Training or Testing (20% of patients are held
out), so test accuracy is not inflated by near-identical slices of one patient.
The conversion itself lives in nifti_slices.py; this script points it at the
datasets, runs a quick visual check, and zips the result.

Examples
--------
Full run on Kaggle (same defaults as the notebook):
    python build_dataset.py

Fast end-to-end smoke test:
    python build_dataset.py --out_dir /tmp/data_real --slices_per_patient 2 \
        --max_patients 5 --no_zip
"""
import argparse
import glob
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no display needed in CI
import matplotlib.pyplot as plt
from PIL import Image

HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "nifti_slices.py"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--glioma_root", default="/kaggle/input/datasets/vietanh21/brats-2021-task-1-dataset")
    p.add_argument("--meningioma_root", default="/kaggle/input/datasets/dukhailly/meningioma")
    p.add_argument("--out_dir", default="/kaggle/working/data_real")
    p.add_argument("--slices_per_patient", type=int, default=8)
    p.add_argument("--max_patients", type=int, default=400)
    p.add_argument("--no_ixi", action="store_true", help="skip IXI healthy controls (no internet needed)")
    p.add_argument("--all_brats", action="store_true", help="use every BraTS case, not just UCSF-PDGM")
    p.add_argument("--notumor_from_tumor_patients", action="store_true",
                   help="also add tumor-free slices from glioma/meningioma scans (see nifti_slices.py)")
    p.add_argument("--list_inputs", action="store_true", help="print the /kaggle/input folder tree and continue")
    p.add_argument("--no_plot", action="store_true", help="skip the visual check image")
    p.add_argument("--no_zip", action="store_true", help="skip creating data_real.zip")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def list_inputs(root="/kaggle/input"):
    """Print a folder tree of the attached datasets (debug helper)."""
    for cur, dirs, files in os.walk(root):
        depth = cur.count(os.sep) - 2
        if depth > 3:
            continue
        print("  " * depth + os.path.basename(cur) + f"/  ({len(dirs)} folders, {len(files)} files)")


def build_slices(args):
    """Run nifti_slices.py to create the Training/Testing slice folders."""
    if not SCRIPT.exists():
        sys.exit(f"nifti_slices.py not found next to this script: {SCRIPT}")
    out_dir = Path(args.out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)

    cmd = [
        sys.executable, str(SCRIPT),
        "--out_dir", str(out_dir),
        "--glioma_root", args.glioma_root,
        "--meningioma_root", args.meningioma_root,
        "--slices_per_patient", str(args.slices_per_patient),
        "--max_patients", str(args.max_patients),
    ]
    if not args.no_ixi:
        cmd.append("--ixi")
    if args.all_brats:
        cmd.append("--all_brats")
    if args.notumor_from_tumor_patients:
        cmd.append("--notumor_from_tumor_patients")

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)  # raises if the conversion fails


def check_output(out_dir):
    """Fail loudly if the expected folder layout or images are missing."""
    for split in ("Training", "Testing"):
        split_dir = Path(out_dir) / split
        if not split_dir.is_dir():
            sys.exit(f"Missing expected folder: {split_dir}")
        n = len(glob.glob(str(split_dir / "*" / "*.jpg")))
        if n == 0:
            sys.exit(f"No .jpg slices found under {split_dir}")
        print(f"{split}: {n} slices")


def save_preview(out_dir, n_per_class=6):
    """Save a grid of random training slices per class to <out_dir>/preview.png."""
    classes = sorted(os.listdir(f"{out_dir}/Training"))
    fig, axes = plt.subplots(len(classes), n_per_class, figsize=(15, 2.7 * len(classes)), squeeze=False)
    for row, cls in enumerate(classes):
        files = glob.glob(f"{out_dir}/Training/{cls}/*.jpg")
        for col, f in enumerate(random.sample(files, min(n_per_class, len(files)))):
            axes[row, col].imshow(Image.open(f), cmap="gray")
            axes[row, col].set_title(f"{cls}\n{os.path.basename(f)[:22]}", fontsize=7)
        for ax in axes[row]:
            ax.axis("off")
    plt.tight_layout()
    path = Path(out_dir) / "preview.png"
    plt.savefig(path, dpi=100)
    plt.close(fig)
    print(f"Saved {path}")


def zip_output(out_dir):
    """Zip the dataset next to out_dir for download / upload to Drive."""
    out_dir = Path(out_dir)
    archive = shutil.make_archive(str(out_dir), "zip", out_dir)
    print(f"Saved {archive}")


def main():
    args = parse_args()
    random.seed(args.seed)
    if args.list_inputs:
        list_inputs()
    build_slices(args)
    check_output(args.out_dir)
    if not args.no_plot:
        save_preview(args.out_dir)
    if not args.no_zip:
        zip_output(args.out_dir)


if __name__ == "__main__":
    main()