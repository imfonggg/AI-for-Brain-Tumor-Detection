"""
Turn MRI scans from several sources into 2D JPEG slices for training and testing.

Output layout (what data_loader.py reads):
    <out_dir>/Training/<class_name>/<patientID>_<plane><index>.jpg
    <out_dir>/Testing/<class_name>/<patientID>_<plane><index>.jpg
    <out_dir>/manifest.csv     one row per slice: path, split, class, source, patient, plane

<plane> is z (axial), y (coronal) or x (sagittal). Every patient goes entirely
into Training/ or entirely into Testing/, and data_loader.py also splits
train/val by patient. manifest.csv records which SOURCE dataset each slice came
from; shortcut_checks.py and loso.py use it.

Sources (each flag adds one source; a class can have several):
    glioma      --glioma_root       BraTS 2021 (only UCSF-PDGM cases unless --all_brats)
    meningioma  --meningioma_root   BraTS-style meningioma set (e.g. BraTS 2023 MEN)
    glioma +    --figshare_root     Figshare brain tumor dataset (Cheng et al., 3,064
    meningioma                      T1 post-contrast 2D slices in .mat files, 233 patients;
    (+pituitary)                    Kaggle mirror ashkhagan/figshare-brain-tumor-dataset)
    metastasis  --metastasis_dir / --download_metastasis
                                    Figshare brain metastasis NIfTI set (*_img / *_msk)
    pituitary   --pituitary_root    Pituitary NET MRI dataset (Figshare 10.6084/m9.figshare.27894084)
    notumor     --ixi / --ixi_dir   IXI healthy controls, T1 (three hospitals: Guys, HH, IOP)
    notumor     --notumor_from_tumor_patients
                                    tumor-free slices from the BraTS patients (same scanners)

Why the harmonized preprocessing (the default, see harmonize.py): when each
class comes from its own dataset, the dataset's "fingerprint" (resolution,
skull or no skull, brightness scale, field of view) identifies the class and the
models learn that instead of the tumor. Every source now goes through the same
reorientation, 1 mm resampling, skull stripping, per-volume intensity
normalization and fixed field of view. --preprocessing legacy reproduces the old
per-slice percentile scaling for comparison.

Usage (Kaggle, after attaching the datasets):
    pip install hd-bet          # for --skullstrip hd-bet (GPU strongly recommended)
    python nifti_slices.py --out_dir data_real \\
        --glioma_root /kaggle/input/brats-2021-task-1-dataset \\
        --meningioma_root /kaggle/input/meningioma \\
        --figshare_root /kaggle/input/figshare-brain-tumor-dataset \\
        --download_metastasis --ixi \\
        --skullstrip hd-bet --planes axial coronal sagittal
"""

import argparse
import csv
import functools
import glob
import hashlib
import math
import os
import re
import shutil
import tarfile
import time
import zipfile
from collections import defaultdict

import numpy as np
import nibabel as nib
from PIL import Image

import harmonize as hz

IMG_SIZE = 256            # data_loader.py resizes to 224 afterwards
SLICES_PER_PATIENT = 8
MAX_PATIENTS = 400        # per class and per source; keeps classes roughly balanced
TEST_FRACTION = 0.2
SPLIT_SEED = 42
RADIOLOGICAL = True       # patient's right on image left (typical clinical view)
PROGRESS_EVERY = 25       # print a progress line every N patients
MANIFEST = "manifest.csv"
MANIFEST_FIELDS = ["path", "split", "class", "source", "patient", "plane"]

# Flush every line so progress shows up live in notebooks and piped logs
print = functools.partial(print, flush=True)

UCSF_CSV_URL = "https://www.cancerimagingarchive.net/wp-content/uploads/UCSF-PDGM-metadata_v5.csv"
IXI_T1_URL = "http://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI-T1.tar"
METASTASIS_IMAGES_URL = "https://ndownloader.figshare.com/files/39604831"   # NIFTIs_Images.zip (~8.5 GB)
METASTASIS_MASKS_URL = "https://ndownloader.figshare.com/files/39603652"    # NIFTIs_Masks.zip (~12 MB)
FIGSHARE_LABELS = {1: "meningioma", 2: "glioma", 3: "pituitary"}

# Manifest rows collected during the run, written by write_manifest()
ROWS = []


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


def record(out_dir, path, class_name, source, patient, plane):
    rel = os.path.relpath(path, out_dir)
    ROWS.append({"path": rel.replace(os.sep, "/"), "split": rel.split(os.sep)[0],
                 "class": class_name, "source": source, "patient": patient, "plane": plane})


def write_manifest(out_dir, rows=None):
    """Append rows (default: everything recorded so far) to <out_dir>/manifest.csv."""
    rows = ROWS if rows is None else rows
    path = os.path.join(out_dir, MANIFEST)
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        if new:
            w.writeheader()
        w.writerows(rows)
    rows.clear()
    return path


# ---------- Volume helpers (legacy preprocessing, kept for comparison) ----------
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
    """LEGACY per-slice intensity normalization (1st-99th percentile of non-zero voxels).
    Returns None for near-empty slices (skull top/bottom, background)."""
    nz = sl[sl > 0]
    if nz.size < min_brain_voxels:
        return None
    lo, hi = np.percentile(nz, [1, 99])
    sl = np.clip((sl - lo) / (hi - lo + 1e-6), 0, 1)
    return (sl * 255).astype(np.uint8)


def save_axial(vol, z, out_path, min_brain_voxels):
    """LEGACY: save one axial slice with per-slice scaling."""
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


# ---------- One code path for every 3D source ----------
def harmonize_config(args):
    return hz.HarmonizeConfig(skullstrip=args.skullstrip, intensity=args.intensity,
                              n4=args.n4, fov_mm=args.fov_mm, img_size=IMG_SIZE,
                              radiological=RADIOLOGICAL)


_warned_unstripped = set()


def load_case(image_path, mask_path, args, already_stripped, source):
    """Legacy: raw RAS arrays. Harmonized: a harmonize.Prepared volume."""
    if args.preprocessing == "legacy":
        return {"vol": load_ras(image_path),
                "seg": load_ras(mask_path) if mask_path else None}
    prep = hz.prepare_volume(resolve_nifti(image_path),
                             resolve_nifti(mask_path) if mask_path else None,
                             harmonize_config(args), already_stripped, workdir=args.tmp_dir)
    if not prep.stripped and source not in _warned_unstripped:
        _warned_unstripped.add(source)
        print(f"  WARNING: {source} still has its skull (--skullstrip none). Other sources "
              f"are skull-stripped, so this is a shortcut. Use --skullstrip hd-bet.")
    return prep


def per_plane(n, args):
    return max(1, math.ceil(n / len(args.planes)))


def pick(case, kind, n, args):
    """[(plane, index), ...] for kind = 'tumor', 'tumor_free' or 'healthy'."""
    if isinstance(case, dict):   # legacy, axial only
        vol, seg = case["vol"], case["seg"]
        if kind == "tumor":
            zs = pick_tumor_slices(seg, n)
        elif kind == "tumor_free":
            zs = pick_tumor_free_slices(vol, seg, n)
        else:
            zs = list((np.linspace(0.40, 0.65, n) * vol.shape[2]).astype(int))
        return [("axial", int(z)) for z in zs]
    k = per_plane(n, args)
    picks = []
    for plane in args.planes:
        if kind == "tumor":
            idx = hz.tumor_slices(case.seg, case.brain, plane, k)
        elif kind == "tumor_free":
            idx = hz.tumor_free_slices(case.seg, case.brain, plane, k)
        else:
            lo, hi = (0.35, 0.75) if plane == "axial" else (0.30, 0.70)
            idx = hz.brain_range_slices(case.brain, plane, k, lo, hi)
        picks += [(plane, int(i)) for i in idx]
    return picks


def write_slices(case, picks, pid, class_name, source, args, prefix=""):
    """Save the picked slices of one case; returns how many were written."""
    out = class_dir(args.out_dir, pid, class_name, args)
    saved = 0
    cfg = None if isinstance(case, dict) else harmonize_config(args)
    for plane, i in picks:
        path = f"{out}/{pid}_{hz.PLANE_TAG[plane]}{prefix}{i:03d}.jpg"
        if cfg is None:
            ok = save_axial(case["vol"], i, path, args.min_brain_voxels)
        else:
            brain2d = hz.take_slice(case.brain, plane, i)
            ok = brain2d.sum() >= args.min_brain_voxels
            if ok:
                hz.render_slice(hz.take_slice(case.norm, plane, i), brain2d, plane,
                                cfg).save(path, quality=95)
        if ok:
            record(args.out_dir, path, class_name, source, pid, plane)
            saved += 1
    return saved


# ---------- BraTS-style datasets (glioma, meningioma, pituitary) ----------
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


# BraTS 2021 names T1-contrast "_t1ce", BraTS 2023+ uses "-t1c"; native T1 is "_t1" / "-t1n"
BRATS_IMAGE = ["*t1ce.nii*", "*t1c.nii*"]
BRATS_IMAGE_T1 = ["*_t1.nii*", "*-t1n.nii*"]
BRATS_MASK = ["*seg.nii*"]
# Pituitary NET dataset: <pid>_T1.nii.gz (post-contrast) and <pid>_tumour.nii.gz
PITUITARY_IMAGE = ["*_T1.nii*"]
PITUITARY_MASK = ["*_tumour.nii*"]


def process_brats(root, class_name, args, source, keep_ids=None,
                  image_patterns=BRATS_IMAGE, mask_patterns=BRATS_MASK,
                  already_stripped=True):
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
    print(f"{class_name} ({source}): found {len(cases)} cases with segmentations")
    target = min(len(cases), args.max_patients)
    start = time.time()
    done = 0
    for case_dir in cases:
        if done >= args.max_patients:
            break
        image = first_match(case_dir, image_patterns)
        seg = first_match(case_dir, mask_patterns)
        if not (image and seg):
            continue
        try:
            case = load_case(image, seg, args, already_stripped, source)
        except Exception as e:
            print(f"  skipping {os.path.basename(case_dir)}: {e}")
            continue
        pid = os.path.basename(case_dir)
        if keep_ids is not None:
            pid = keep_ids[pid]
        saved = write_slices(case, pick(case, "tumor", args.slices_per_patient, args),
                             pid, class_name, source, args)
        if args.notumor_from_tumor_patients:
            # Same patient, same split -- only the class folder differs.
            write_slices(case, pick(case, "tumor_free", args.slices_per_patient // 2, args),
                         pid, "notumor", source, args)
        if saved:
            done += 1
            if done % PROGRESS_EVERY == 0:
                print(f"  {class_name}: {done}/{target} patients ({time.time() - start:.0f}s)")
    print(f"{class_name} ({source}): saved slices from {done} patients")


# ---------- IXI healthy controls ----------
def ixi_source(name):
    """IXI002-Guys-0828-T1.nii.gz -> 'ixi-guys'. IXI has three hospitals/scanners."""
    parts = name.split("-")
    return f"ixi-{parts[1].lower()}" if len(parts) > 2 else "ixi"


def process_ixi_file(path, name, args):
    pid = name.replace("-T1.nii.gz", "")
    source = ixi_source(name)
    case = load_case(path, None, args, already_stripped=False, source=source)
    # No mask for healthy brains: slices from the mid brain (the levels where
    # tumors are sampled), chosen the same way for every source.
    return write_slices(case, pick(case, "healthy", args.slices_per_patient, args),
                        pid, "notumor", source, args)


def process_ixi(args, hospitals=None):
    """Stream IXI T1 from the IXI server (nothing large kept on disk)."""
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
                try:
                    saved = process_ixi_file(path, name, args)
                except Exception as e:
                    print(f"  skipping {name}: {e}")
                    saved = 0
                os.remove(path)
                if saved:
                    done += 1
                    if done % PROGRESS_EVERY == 0:
                        print(f"  notumor (IXI): {done}/{args.max_patients} patients "
                              f"({time.time() - start:.0f}s)")
    print(f"notumor (IXI): saved slices from {done} patients")


def process_ixi_dir(root, args, hospitals=None):
    """IXI T1 files already on disk (e.g. an extracted IXI-T1.tar)."""
    files = sorted(glob.glob(f"{root}/**/IXI*-T1.nii.gz", recursive=True))
    if hospitals:
        files = [f for f in files if os.path.basename(f).split("-")[1] in hospitals]
    print(f"notumor (IXI): {len(files)} local T1 files")
    done = 0
    for f in files:
        if done >= args.max_patients:
            break
        try:
            done += bool(process_ixi_file(f, os.path.basename(f), args))
        except Exception as e:
            print(f"  skipping {os.path.basename(f)}: {e}")
    print(f"notumor (IXI): saved slices from {done} patients")


# ---------- Figshare brain metastasis set (NIfTI, *_img / *_msk) ----------
def _download(url, path):
    import requests
    if os.path.exists(path):
        print(f"already downloaded: {path}")
        return
    print(f"downloading {url} -> {path}")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done, next_report = 0, 0
        with open(path + ".part", "wb") as f:
            for chunk in r.iter_content(chunk_size=8 << 20):
                f.write(chunk)
                done += len(chunk)
                if done >= next_report:
                    print(f"  {done / 1e9:.1f} / {total / 1e9:.1f} GB")
                    next_report += 500 << 20
    os.rename(path + ".part", path)


def _scan_key(name):
    """'.../X_img.nii.gz' and '.../X_msk.nii.gz' -> 'X' so images and masks pair up."""
    return re.sub(r"_(img|msk)\.nii(\.gz)?$", "", os.path.basename(name))


def download_metastasis(work_dir):
    """Download the Figshare metastasis set and extract only images that have a mask."""
    os.makedirs(work_dir, exist_ok=True)
    masks_zip, images_zip = f"{work_dir}/masks.zip", f"{work_dir}/images.zip"
    _download(METASTASIS_MASKS_URL, masks_zip)
    with zipfile.ZipFile(masks_zip) as z:
        z.extractall(f"{work_dir}/masks")
    masks = {_scan_key(p) for p in glob.glob(f"{work_dir}/masks/**/*_msk.nii*", recursive=True)}
    if not glob.glob(f"{work_dir}/images/**/*_img.nii*", recursive=True):
        _download(METASTASIS_IMAGES_URL, images_zip)
        with zipfile.ZipFile(images_zip) as z:
            members = [m for m in z.namelist()
                       if re.search(r"_img\.nii(\.gz)?$", m) and _scan_key(m) in masks]
            print(f"extracting {len(members)} matching images")
            for m in members:
                z.extract(m, f"{work_dir}/images")
        os.remove(images_zip)
    return work_dir


def process_metastasis(root, args, class_name="metastasis", source="figshare-mets",
                       studies_per_patient=1):
    masks = {_scan_key(p): p for p in glob.glob(f"{root}/**/*_msk.nii*", recursive=True)}
    images = sorted(p for p in glob.glob(f"{root}/**/*_img.nii*", recursive=True)
                    if _scan_key(p) in masks)
    if not images:
        raise RuntimeError(f"No *_img.nii files with a matching *_msk.nii under {root}")
    base = os.path.commonpath([os.path.dirname(i) for i in images])   # skips wrapper folders

    def patient_of(img):
        parts = os.path.relpath(img, base).replace("\\", "/").split("/")
        pid = parts[0] if len(parts) > 1 else parts[0].split("_")[0]
        return re.sub(r"[^A-Za-z0-9-]", "-", pid)

    by_patient = defaultdict(list)
    for img in images:
        by_patient[patient_of(img)].append(img)
    print(f"{class_name} ({source}): {len(by_patient)} patients")
    done = 0
    for pid, scans in sorted(by_patient.items()):
        if done >= args.max_patients:
            break
        saved = 0
        for k, img in enumerate(scans[:studies_per_patient]):
            try:
                case = load_case(img, masks[_scan_key(img)], args, already_stripped=False,
                                 source=source)
            except Exception as e:
                print(f"  skipping {os.path.basename(img)}: {e}")
                continue
            # prefix k keeps '<pid>_<plane><digits>' so every scan groups by patient
            saved += write_slices(case, pick(case, "tumor", args.slices_per_patient, args),
                                  pid, class_name, source, args, prefix=str(k))
        if saved:
            done += 1
            if done % PROGRESS_EVERY == 0:
                print(f"  {class_name}: {done}/{len(by_patient)} patients")
    print(f"{class_name} ({source}): saved slices from {done} patients")


# ---------- Figshare brain tumor dataset (Cheng et al.), 2D .mat slices ----------
def read_figshare_mat(path):
    """Return (label, pid, image, tumor_mask) from one Figshare .mat file, or None.
    The original files are MATLAB v7.3 (HDF5); older v5 copies also exist."""
    try:
        import h5py
        with h5py.File(path, "r") as f:
            if "cjdata" not in f:
                return None
            d = f["cjdata"]
            label = int(np.asarray(d["label"]).ravel()[0])
            pid = "".join(chr(int(c)) for c in np.asarray(d["PID"]).ravel() if int(c) > 0)
            image = np.asarray(d["image"], np.float32).T       # MATLAB is column-major
            mask = np.asarray(d["tumorMask"]).T > 0
            return label, pid.strip(), image, mask
    except OSError:
        from scipy.io import loadmat
        m = loadmat(path, squeeze_me=True, struct_as_record=False)
        if "cjdata" not in m:
            return None
        d = m["cjdata"]
        pid = d.PID if isinstance(d.PID, str) else "".join(map(chr, np.ravel(d.PID)))
        return int(d.label), pid.strip(), np.asarray(d.image, np.float32), np.asarray(d.tumorMask) > 0


def process_figshare(root, args, classes=("glioma", "meningioma"), source="figshare-cheng"):
    """Figshare 2D slices. These are single slices (mixed axial/coronal/sagittal),
    not volumes, so they get the 2D version of the same harmonization: resample
    to 1 mm pixels, rough skull strip, per-PATIENT intensity normalization over
    brain pixels, the same fixed field of view and gray mapping."""
    files = sorted(f for f in glob.glob(f"{root}/**/*.mat", recursive=True)
                   if os.path.basename(f).lower() != "cvind.mat")
    by_patient = defaultdict(list)
    for f in files:
        item = read_figshare_mat(f)
        if item is None:
            continue
        label, pid, image, _ = item
        cls = FIGSHARE_LABELS.get(label)
        if cls in classes:
            by_patient[(cls, pid)].append(f)
    print(f"figshare ({source}): {len(files)} .mat files, "
          f"{len(by_patient)} patient/class groups for {list(classes)}")
    cfg = harmonize_config(args)
    counts = defaultdict(int)
    for (cls, raw_pid), paths in sorted(by_patient.items()):
        if counts[cls] >= args.max_patients:
            continue
        paths = sorted(paths, key=lambda p: int(re.sub(r"\D", "", os.path.basename(p)) or 0))
        paths = evenly_spaced(np.array(paths), args.slices_per_patient)
        pid = f"figshare-{re.sub(r'[^A-Za-z0-9-]', '-', raw_pid)}"
        out = class_dir(args.out_dir, pid, cls, args)
        slices = []
        for p in paths:
            _, _, image, _ = read_figshare_mat(p)
            idx = int(re.sub(r"\D", "", os.path.basename(p)) or 0)
            slices.append((idx, image))
        saved = 0
        if args.preprocessing == "legacy":
            for idx, image in slices:
                u8 = to_uint8(image, args.min_brain_voxels)
                if u8 is None:
                    continue
                path = f"{out}/{pid}_z{idx:04d}.jpg"
                Image.fromarray(u8).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR).save(path, quality=95)
                record(args.out_dir, path, cls, source, pid, "mixed")
                saved += 1
        else:
            prepared = []
            for idx, image in slices:
                # Figshare images are 512x512 (a few 256x256) over ~250 mm
                pixel_mm = args.figshare_pixel_mm * 512 / image.shape[0]
                img1 = hz.ndimage.zoom(np.clip(image, 0, None), pixel_mm / cfg.spacing_mm, order=1)
                brain = (hz.heuristic_brain_mask(img1, cfg.spacing_mm)
                         if cfg.skullstrip != "none" else hz.clean_mask(img1 > hz.otsu_threshold(img1[img1 > 0])))
                prepared.append((idx, img1 * brain, brain))
            # one set of normalization statistics per patient (the 2D stand-in for per-volume)
            pooled = np.concatenate([a[b] for _, a, b in prepared]) if prepared else np.array([])
            normalizer = hz.fit_normalizer(pooled, cfg.intensity)
            for idx, arr, brain in prepared:
                if brain.sum() < args.min_brain_voxels:
                    continue
                norm = hz.normalize_intensity(arr, brain, normalizer=normalizer)
                path = f"{out}/{pid}_z{idx:04d}.jpg"
                hz.render_slice(norm, brain, "axial", cfg, already_oriented=True).save(path, quality=95)
                record(args.out_dir, path, cls, source, pid, "mixed")
                saved += 1
        if saved:
            counts[cls] += 1
    for cls in classes:
        print(f"{cls} ({source}): saved slices from {counts[cls]} patients")


def summarize(out_dir):
    """Print image and patient counts per split/class, and per source if a manifest exists."""
    for split in ("Training", "Testing"):
        split_dir = os.path.join(out_dir, split)
        if not os.path.isdir(split_dir):
            continue
        print(f"\n{split}:")
        for cls in sorted(os.listdir(split_dir)):
            files = os.listdir(os.path.join(split_dir, cls))
            patients = {re.sub(r"_[xyz]\d+\.jpg$", "", f) for f in files}
            print(f"  {cls:<12} {len(files):>6} images from {len(patients):>4} patients")
    manifest = os.path.join(out_dir, MANIFEST)
    if os.path.exists(manifest):
        counts = defaultdict(lambda: defaultdict(int))
        with open(manifest) as f:
            for row in csv.DictReader(f):
                counts[row["class"]][row["source"]] += 1
        print("\nSources per class (a class with one source can still be told apart by its dataset):")
        for cls in sorted(counts):
            srcs = ", ".join(f"{s} {n}" for s, n in sorted(counts[cls].items()))
            flag = "" if len(counts[cls]) > 1 else "   <- single source"
            print(f"  {cls:<12} {srcs}{flag}")


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
    parser.add_argument("--sequence", choices=["t1c", "t1"], default="t1c",
                        help="BraTS sequence: t1c (post-contrast, default) or t1 (native, "
                             "the same contrast as the IXI healthy scans)")
    parser.add_argument("--figshare_root", default=None,
                        help="Figshare brain tumor dataset folder with the .mat files "
                             "(second source for glioma and meningioma)")
    parser.add_argument("--figshare_classes", nargs="+", default=["glioma", "meningioma"],
                        choices=sorted(FIGSHARE_LABELS.values()))
    parser.add_argument("--figshare_pixel_mm", type=float, default=0.49,
                        help="Pixel size of the 512x512 Figshare slices")
    parser.add_argument("--metastasis_dir", default=None,
                        help="Folder with the Figshare metastasis *_img/*_msk NIfTI files")
    parser.add_argument("--download_metastasis", action="store_true",
                        help="Download the Figshare metastasis set (~8.5 GB) into tmp_dir")
    parser.add_argument("--pituitary_root", default=None,
                        help="Pituitary NET dataset folder (<pid>_T1 + <pid>_tumour files)")
    parser.add_argument("--ixi", action="store_true",
                        help="Stream IXI T1 healthy controls (needs internet)")
    parser.add_argument("--ixi_dir", default=None,
                        help="Use IXI T1 files already on disk instead of streaming")
    parser.add_argument("--ixi_hospitals", nargs="*", default=None,
                        help="Only these IXI sites, e.g. HH Guys IOP")
    parser.add_argument("--notumor_from_tumor_patients", action="store_true",
                        help="Also save tumor-free slices from glioma/meningioma "
                             "patients as notumor (same scanners as the tumor classes)")
    # Harmonization
    parser.add_argument("--preprocessing", choices=["harmonized", "legacy"], default="harmonized",
                        help="legacy = old per-slice scaling, no resampling or skull strip")
    parser.add_argument("--skullstrip", default="heuristic",
                        help="Skull stripping for sources that keep the skull (IXI, metastasis, "
                             "pituitary, Figshare): hd-bet (best; pip install hd-bet), synthstrip, "
                             "heuristic (rough, no install), none, or a command template with "
                             "{input} {output} {device}. BraTS sets are already stripped.")
    parser.add_argument("--intensity", choices=["zscore", "histmatch"], default="zscore",
                        help="Per-volume intensity normalization over brain voxels")
    parser.add_argument("--n4", action="store_true",
                        help="N4 bias-field correction (pip install SimpleITK; slow on CPU)")
    parser.add_argument("--planes", nargs="+", default=["axial"],
                        choices=list(hz.PLANE_AXIS),
                        help="Planes to slice. Use all three when adding the Figshare slices, "
                             "which mix axial, coronal and sagittal views")
    parser.add_argument("--fov_mm", type=int, default=hz.FOV_MM)
    parser.add_argument("--slices_per_patient", type=int, default=SLICES_PER_PATIENT,
                        help="Per patient, split over the chosen planes")
    parser.add_argument("--max_patients", type=int, default=MAX_PATIENTS,
                        help="Per class and per source")
    parser.add_argument("--min_brain_voxels", type=int, default=500,
                        help="Skip slices with fewer brain voxels than this")
    parser.add_argument("--test_fraction", type=float, default=TEST_FRACTION)
    parser.add_argument("--seed", type=int, default=SPLIT_SEED)
    parser.add_argument("--tmp_dir", default="/tmp")
    args = parser.parse_args()

    if args.preprocessing == "legacy" and args.planes != ["axial"]:
        parser.error("--preprocessing legacy only supports --planes axial")
    os.makedirs(args.out_dir, exist_ok=True)
    if os.path.exists(os.path.join(args.out_dir, MANIFEST)):
        os.remove(os.path.join(args.out_dir, MANIFEST))
    brats_image = BRATS_IMAGE if args.sequence == "t1c" else BRATS_IMAGE_T1
    print(f"preprocessing={args.preprocessing} skullstrip={args.skullstrip} "
          f"intensity={args.intensity} planes={args.planes}")

    if args.glioma_root:
        ucsf_map = None if args.all_brats else load_ucsf_mapping()
        if ucsf_map is not None:
            print(f"UCSF-PDGM cases that are in BraTS 2021: {len(ucsf_map)}")
        process_brats(args.glioma_root, "glioma", args,
                      source="brats-ucsf" if ucsf_map is not None else "brats2021",
                      keep_ids=ucsf_map, image_patterns=brats_image)
    if args.meningioma_root:
        process_brats(args.meningioma_root, "meningioma", args, source="brats-men",
                      image_patterns=brats_image)
    if args.figshare_root:
        process_figshare(args.figshare_root, args, classes=tuple(args.figshare_classes))
    if args.download_metastasis and not args.metastasis_dir:
        args.metastasis_dir = download_metastasis(os.path.join(args.tmp_dir, "metastasis"))
    if args.metastasis_dir:
        process_metastasis(args.metastasis_dir, args)
    if args.pituitary_root:
        process_brats(args.pituitary_root, "pituitary", args, source="pituitary-net",
                      image_patterns=PITUITARY_IMAGE, mask_patterns=PITUITARY_MASK,
                      already_stripped=False)
    if args.ixi_dir:
        process_ixi_dir(args.ixi_dir, args, hospitals=args.ixi_hospitals)
    elif args.ixi:
        process_ixi(args, hospitals=args.ixi_hospitals)

    print(f"\nmanifest: {write_manifest(args.out_dir)}")
    summarize(args.out_dir)


if __name__ == "__main__":
    main()
