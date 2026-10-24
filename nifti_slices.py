"""
Turn 3D NIfTI MRI volumes into 2D axial JPEG slices for training and testing.

Output layout (what data_loader.py reads):
    <out_dir>/Training/<class_name>/<patientID>_z###.jpg
    <out_dir>/Testing/<class_name>/<patientID>_z###.jpg

Every patient goes entirely into Training/ or entirely into Testing/. Slices
from the same patient look almost identical, so splitting by slice would leak
patients across the split and inflate test accuracy. The patient ID is kept in
every filename so data_loader.py can also split train/val by patient.

Sources currently supported:
    glioma      BraTS 2021 (optionally only the UCSF-PDGM cases), T1 post-contrast
    meningioma  BraTS-style meningioma set (e.g. BraTS 2023 MEN), T1 post-contrast
    pituitary   Pituitary NET MRI dataset (Figshare 10.6084/m9.figshare.27894084),
                T1 post-contrast with tumour masks
    notumor     IXI healthy controls, T1, streamed from the IXI server

Usage (Kaggle or Colab, after attaching the datasets):
    python nifti_slices.py --out_dir data_real \
        --glioma_root /kaggle/input/brats-2021-task-1-dataset \
        --meningioma_root /kaggle/input/meningioma \
        --ixi

Leave out a source's flag to skip that class.

Shortcut risk: glioma/meningioma slices are T1 post-contrast from BraTS
scanners, while IXI is plain T1 from other hospitals. A model can score well by
telling contrast/scanner apart instead of tumor/no tumor.
--notumor_from_tumor_patients adds tumor-free T1 post-contrast slices (far from
the tumor) from the same BraTS patients as extra notumor examples, which makes
that shortcut less useful. Those brains still have a tumor elsewhere, so check
the results with and without it. BraTS volumes are also skull-stripped while IXI
and the pituitary set are not, which is a second shortcut of the same kind.
"""

import argparse
import functools
import glob
import hashlib
import os
import shutil
import tarfile
import time

import numpy as np
import nibabel as nib
from PIL import Image

IMG_SIZE = 256            # data_loader.py resizes to 224 afterwards
SLICES_PER_PATIENT = 8
MAX_PATIENTS = 400        # per class; keeps classes roughly balanced
TEST_FRACTION = 0.2
SPLIT_SEED = 42
RADIOLOGICAL = True       # patient's right on image left (typical clinical view)
PROGRESS_EVERY = 25      # print a progress line every N patients

# Flush every line so progress shows up live in notebooks and piped logs
print = functools.partial(print, flush=True)

UCSF_CSV_URL = "https://www.cancerimagingarchive.net/wp-content/uploads/UCSF-PDGM-metadata_v5.csv"
IXI_T1_URL = "http://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI-T1.tar"


# ---------- Patient-level split ----------
def split_for(patient_id, test_fraction=TEST_FRACTION, seed=SPLIT_SEED):
    """Deterministically assign a patient to Training or Testing.

    Hash-based rather than shuffle-based so it works while streaming (IXI)
    without knowing the full patient list, and gives the same answer on every
    run with the same seed.
    """
    digest = hashlib.md5(f"{seed}:{patient_id}".encode()).hexdigest()
    return "Testing" if int(digest[:8], 16) / 0xFFFFFFFF < test_fraction else "Training"


def class_dir(out_dir, patient_id, class_name, args):
    path = os.path.join(out_dir, split_for(patient_id, args.test_fraction, args.seed),
                        class_name)
    os.makedirs(path, exist_ok=True)
    return path


# ---------- Volume helpers ----------
def resolve_nifti(path):
    """Some Kaggle mirrors store each scan as a FOLDER named like 'BraTS2021_00000_t1ce.nii'
    with the real .nii file inside. Return the real file in that case."""
    if os.path.isdir(path):
        inner = sorted(glob.glob(os.path.join(path, "**", "*.nii*"), recursive=True))
        inner = [f for f in inner if os.path.isfile(f)]
        if not inner:
            raise FileNotFoundError(f"No NIfTI file inside folder {path}")
        return inner[0]
    return path


def load_ras(path):
    """Load a NIfTI volume reoriented to RAS, so axis 2 is axial (inferior -> superior)."""
    img = nib.as_closest_canonical(nib.load(resolve_nifti(path)))
    return np.asarray(img.dataobj, dtype=np.float32)


def to_uint8(sl, min_brain_voxels):
    """Per-slice intensity normalization (1st-99th percentile of non-zero voxels).
    Returns None for near-empty slices (skull top/bottom, background)."""
    nz = sl[sl > 0]
    if nz.size < min_brain_voxels:
        return None
    lo, hi = np.percentile(nz, [1, 99])
    sl = np.clip((sl - lo) / (hi - lo + 1e-6), 0, 1)
    return (sl * 255).astype(np.uint8)


def save_axial(vol, z, out_path, min_brain_voxels):
    sl = to_uint8(vol[:, :, z], min_brain_voxels)
    if sl is None:
        return False
    sl = np.rot90(sl)              # anterior at the top
    if RADIOLOGICAL:
        sl = np.fliplr(sl)
    Image.fromarray(sl).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR).save(out_path, quality=95)
    return True


def evenly_spaced(candidates, n):
    if len(candidates) <= n:
        return list(candidates)
    return list(candidates[np.linspace(0, len(candidates) - 1, n).round().astype(int)])


def pick_tumor_slices(seg, n):
    """Pick n evenly spaced axial slices among those with >= 50% of the max tumor area.
    Uses tumor core (excludes edema, label 2 in BraTS) when present."""
    core = (seg > 0) & (seg != 2)
    mask = core if core.any() else (seg > 0)
    area = mask.sum(axis=(0, 1))
    if area.max() == 0:
        return []
    return evenly_spaced(np.where(area >= 0.5 * area.max())[0], n)


def pick_tumor_free_slices(vol, seg, n, margin=10):
    """Slices with no tumor label at least `margin` slices away from any tumor,
    restricted to slices that contain a reasonable amount of brain."""
    has_tumor = (seg > 0).sum(axis=(0, 1)) > 0
    tumor_z = np.where(has_tumor)[0]
    if tumor_z.size == 0:
        return []
    brain_area = (vol > 0).sum(axis=(0, 1))
    z = np.arange(vol.shape[2])
    far = (z < tumor_z.min() - margin) | (z > tumor_z.max() + margin)
    ok = far & (brain_area >= 0.5 * brain_area.max())
    return evenly_spaced(np.where(ok)[0], n)


def first_match(folder, patterns):
    for p in patterns:
        hits = glob.glob(os.path.join(folder, p))
        if hits:
            return hits[0]
    return None


# ---------- BraTS-style datasets (glioma, meningioma) ----------
def load_ucsf_mapping():
    """Return {BraTS2021_xxxxx: UCSF-PDGM-xxx} from the UCSF-PDGM clinical CSV."""
    import pandas as pd
    import io
    import requests
    print("Downloading UCSF-PDGM case list...")
    r = requests.get(UCSF_CSV_URL, timeout=60)
    r.raise_for_status()
    meta = pd.read_csv(io.StringIO(r.text))
    meta = meta[meta["BraTS21 ID"].notna()]
    return dict(zip(meta["BraTS21 ID"].str.strip(), meta["ID"].str.strip()))


# BraTS 2021 names T1-contrast "_t1ce", BraTS 2023+ uses "-t1c"
BRATS_IMAGE = ["*t1ce.nii*", "*t1c.nii*"]
BRATS_MASK = ["*seg.nii*"]
# Pituitary NET dataset: <pid>_T1.nii.gz (post-contrast) and <pid>_tumour.nii.gz
PITUITARY_IMAGE = ["*_T1.nii*"]
PITUITARY_MASK = ["*_tumour.nii*"]


def process_brats(root, class_name, args, keep_ids=None,
                  image_patterns=BRATS_IMAGE, mask_patterns=BRATS_MASK):
    """Datasets with one folder per patient holding an image and a tumor mask.
    keep_ids: optional {folder_name: output_name}; only those cases are used."""
    cases = set()
    for pattern in mask_patterns:
        for p in glob.glob(f"{root}/**/{pattern}", recursive=True):
            d = os.path.dirname(p)
            if ".nii" in os.path.basename(d):   # match was the file inside a '*seg.nii' folder
                d = os.path.dirname(d)
            cases.add(d)
    cases = sorted(cases)
    if keep_ids is not None:
        cases = [c for c in cases if os.path.basename(c) in keep_ids]
    print(f"{class_name}: found {len(cases)} cases with segmentations")
    target = min(len(cases), args.max_patients)
    start = time.time()
    done = 0
    for case in cases:
        if done >= args.max_patients:
            break
        t1c = first_match(case, image_patterns)
        seg = first_match(case, mask_patterns)
        if not (t1c and seg):
            continue
        try:
            vol, mask = load_ras(t1c), load_ras(seg)
        except Exception as e:
            print(f"  skipping {os.path.basename(case)}: {e}")
            continue
        pid = os.path.basename(case)
        if keep_ids is not None:
            pid = keep_ids[pid]
        out = class_dir(args.out_dir, pid, class_name, args)
        zs = pick_tumor_slices(mask, args.slices_per_patient)
        saved = sum(save_axial(vol, z, f"{out}/{pid}_z{z:03d}.jpg", args.min_brain_voxels)
                    for z in zs)
        if args.notumor_from_tumor_patients:
            # Same patient, same split -- only the class folder differs.
            free_out = class_dir(args.out_dir, pid, "notumor", args)
            for z in pick_tumor_free_slices(vol, mask, args.slices_per_patient // 2):
                save_axial(vol, z, f"{free_out}/{pid}_z{z:03d}.jpg", args.min_brain_voxels)
        if saved:
            done += 1
            if done % PROGRESS_EVERY == 0:
                print(f"  {class_name}: {done}/{target} patients ({time.time() - start:.0f}s)")
    print(f"{class_name}: saved slices from {done} patients")


# ---------- IXI healthy controls (streamed, nothing large kept on disk) ----------
def process_ixi(args, hospitals=None):
    tmp = os.path.join(args.tmp_dir, "ixi")
    os.makedirs(tmp, exist_ok=True)
    import requests
    print(f"notumor (IXI): streaming {IXI_T1_URL}")
    start = time.time()
    done = 0
    with requests.get(IXI_T1_URL, stream=True, timeout=120) as r:
        r.raise_for_status()
        with tarfile.open(fileobj=r.raw, mode="r|") as tar:   # read the archive as a stream
            for m in tar:
                if done >= args.max_patients:
                    break                                     # stops the download early
                if not m.name.endswith(".nii.gz"):
                    continue
                name = os.path.basename(m.name)               # e.g. IXI002-Guys-0828-T1.nii.gz
                if hospitals and name.split("-")[1] not in hospitals:
                    continue
                path = os.path.join(tmp, name)
                with open(path, "wb") as f:
                    shutil.copyfileobj(tar.extractfile(m), f)
                vol = load_ras(path)
                # No mask for healthy brains: take slices from the mid brain
                # (roughly the levels where tumors in BraTS are sampled).
                nz = vol.shape[2]
                zs = (np.linspace(0.40, 0.65, args.slices_per_patient) * nz).astype(int)
                pid = name.replace("-T1.nii.gz", "")
                out = class_dir(args.out_dir, pid, "notumor", args)
                saved = sum(save_axial(vol, z, f"{out}/{pid}_z{z:03d}.jpg", args.min_brain_voxels)
                            for z in zs)
                os.remove(path)
                if saved:
                    done += 1
                    if done % PROGRESS_EVERY == 0:
                        print(f"  notumor (IXI): {done}/{args.max_patients} patients "
                              f"({time.time() - start:.0f}s)")
    print(f"notumor (IXI): saved slices from {done} patients")


def summarize(out_dir):
    """Print image and patient counts per split/class."""
    for split in ("Training", "Testing"):
        split_dir = os.path.join(out_dir, split)
        if not os.path.isdir(split_dir):
            continue
        print(f"\n{split}:")
        for cls in sorted(os.listdir(split_dir)):
            files = os.listdir(os.path.join(split_dir, cls))
            patients = {f.rsplit("_z", 1)[0] for f in files}
            print(f"  {cls:<12} {len(files):>6} images from {len(patients):>4} patients")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out_dir", default="data_real")
    parser.add_argument("--glioma_root", default=None,
                        help="BraTS 2021 folder (one subfolder per patient)")
    parser.add_argument("--all_brats", action="store_true",
                        help="Use every BraTS 2021 case instead of only UCSF-PDGM ones")
    parser.add_argument("--meningioma_root", default=None,
                        help="BraTS-style meningioma folder")
    parser.add_argument("--pituitary_root", default=None,
                        help="Pituitary NET dataset folder (<pid>_T1 + <pid>_tumour files)")
    parser.add_argument("--ixi", action="store_true",
                        help="Stream IXI T1 healthy controls (needs internet)")
    parser.add_argument("--ixi_hospitals", nargs="*", default=None,
                        help="Only these IXI sites, e.g. HH Guys IOP")
    parser.add_argument("--notumor_from_tumor_patients", action="store_true",
                        help="Also save tumor-free slices from glioma/meningioma "
                             "patients as notumor (T1 post-contrast, same scanners)")
    parser.add_argument("--slices_per_patient", type=int, default=SLICES_PER_PATIENT)
    parser.add_argument("--max_patients", type=int, default=MAX_PATIENTS,
                        help="Per class and per source")
    parser.add_argument("--min_brain_voxels", type=int, default=500,
                        help="Skip slices with fewer non-zero voxels than this")
    parser.add_argument("--test_fraction", type=float, default=TEST_FRACTION)
    parser.add_argument("--seed", type=int, default=SPLIT_SEED)
    parser.add_argument("--tmp_dir", default="/tmp")
    args = parser.parse_args()

    if args.glioma_root:
        ucsf_map = None if args.all_brats else load_ucsf_mapping()
        if ucsf_map is not None:
            print(f"UCSF-PDGM cases that are in BraTS 2021: {len(ucsf_map)}")
        process_brats(args.glioma_root, "glioma", args, keep_ids=ucsf_map)
    if args.meningioma_root:
        process_brats(args.meningioma_root, "meningioma", args)
    if args.pituitary_root:
        process_brats(args.pituitary_root, "pituitary", args,
                      image_patterns=PITUITARY_IMAGE, mask_patterns=PITUITARY_MASK)
    if args.ixi:
        process_ixi(args, hospitals=args.ixi_hospitals)

    summarize(args.out_dir)


if __name__ == "__main__":
    main()
