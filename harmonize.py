"""
Harmonized preprocessing: make every source look the same before slicing.

Each class used to come from one dataset with its own scanner, resolution,
orientation, skull/no skull and brightness scale, and the models learned to
recognise those "fingerprints" instead of the tumor (see the pixel-shuffle test
in shortcut_checks.py). This module applies the same steps to every volume, no
matter where it came from:

    1. Orientation   reorient to RAS (axis 0 = left->right, 1 = posterior->anterior,
                     2 = inferior->superior)
    2. Resolution    resample to 1 mm isotropic voxels
    3. Skull strip   run HD-BET / SynthStrip (or a rough built-in fallback) on
                     sources that still have a skull; BraTS-style sets already
                     are skull-stripped
    4. Bias field    optional N4 correction (needs SimpleITK)
    5. Intensity     per-volume normalization over brain voxels only:
                       zscore     robust z-score (default)
                       histmatch  piecewise-linear histogram matching of brain
                                  percentiles to a fixed standard-normal reference
    6. Framing       every slice is cut to the same physical field of view
                     (FOV_MM x FOV_MM mm) around the brain, so the size of the
                     brain on screen and the black border do not depend on the
                     source's matrix size

Slices are written as 8-bit images with a FIXED mapping from normalized
intensity to gray level (Z_WINDOW), so the same tissue gets the same gray value
in every source. Background outside the brain is exactly 0.
"""

import os
import shlex
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Optional

import numpy as np
import nibabel as nib
from nibabel.processing import resample_from_to, resample_to_output
from scipy import ndimage

TARGET_SPACING_MM = 1.0
FOV_MM = 200                  # every slice covers 200 x 200 mm around the brain
Z_WINDOW = (-3.0, 4.0)        # normalized intensities mapped to gray 1..255
PLANE_AXIS = {"sagittal": 0, "coronal": 1, "axial": 2}
PLANE_TAG = {"sagittal": "x", "coronal": "y", "axial": "z"}   # used in filenames

# Command templates for external skull-stripping tools. {input} and {output}
# are .nii.gz paths; {device} is "cuda" when a GPU is visible, else "cpu".
SKULLSTRIP_COMMANDS = {
    # HD-BET v2: pip install hd-bet  (downloads its weights on first use)
    "hd-bet": "hd-bet -i {input} -o {output} -device {device} --disable_tta",
    # SynthStrip: FreeSurfer's mri_synthstrip (or the synthstrip docker wrapper)
    "synthstrip": "mri_synthstrip -i {input} -o {output}",
}

# Standard-normal quantiles used as the fixed reference for histmatch
_HM_PERCENTILES = np.array([1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 98, 99], float)
_HM_TARGETS = np.array([-2.326, -2.054, -1.645, -1.282, -0.842, -0.524, -0.253, 0.0,
                        0.253, 0.524, 0.842, 1.282, 1.645, 2.054, 2.326])


@dataclass
class HarmonizeConfig:
    skullstrip: str = "none"          # none | hd-bet | synthstrip | heuristic | a custom command
    intensity: str = "zscore"         # zscore | histmatch
    n4: bool = False
    spacing_mm: float = TARGET_SPACING_MM
    fov_mm: int = FOV_MM
    img_size: int = 256
    radiological: bool = True         # patient's right on image left


@dataclass
class Prepared:
    """A harmonized volume, ready to slice."""
    norm: np.ndarray                  # normalized intensities (0 outside the brain)
    brain: np.ndarray                 # bool brain mask
    seg: Optional[np.ndarray]         # tumor mask on the same grid, or None
    spacing_mm: float
    stripped: bool                    # False if the skull could not be removed


# ---------- loading, orientation, resolution ----------
def load_canonical(path):
    """nibabel image reoriented to RAS."""
    return nib.as_closest_canonical(nib.load(path))


def resample_iso(img, spacing=TARGET_SPACING_MM, order=1):
    """Resample to isotropic voxels (no-op if already within 5%)."""
    zooms = np.asarray(img.header.get_zooms()[:3], float)
    if np.allclose(zooms, spacing, rtol=0.05):
        return img
    out = resample_to_output(img, voxel_sizes=(spacing,) * 3, order=order)
    return nib.as_closest_canonical(out)


# ---------- skull stripping ----------
def _device():
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def run_skullstrip_tool(img, method, workdir=None):
    """Run an external skull-stripping tool on a nibabel image; returns the stripped image."""
    template = SKULLSTRIP_COMMANDS.get(method, method)
    tool = shlex.split(template)[0]
    if shutil.which(tool) is None:
        raise RuntimeError(
            f"Skull-stripping tool '{tool}' is not installed. For HD-BET run "
            f"'pip install hd-bet', or pass --skullstrip heuristic / none.")
    tmp = tempfile.mkdtemp(dir=workdir)
    try:
        src, dst = os.path.join(tmp, "in.nii.gz"), os.path.join(tmp, "out.nii.gz")
        nib.save(img, src)
        cmd = template.format(input=shlex.quote(src), output=shlex.quote(dst), device=_device())
        subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL,
                       stderr=subprocess.PIPE)
        out = nib.load(dst)
        return nib.Nifti1Image(np.asarray(out.dataobj, np.float32), img.affine)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def otsu_threshold(values, bins=256):
    hist, edges = np.histogram(values, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    m0 = np.cumsum(hist * centers) / np.maximum(w0, 1)
    m1 = (np.sum(hist * centers) - np.cumsum(hist * centers)) / np.maximum(w1, 1)
    return centers[np.argmax(w0 * w1 * (m0 - m1) ** 2)]


def _largest_component(mask):
    labels, n = ndimage.label(mask)
    if n == 0:
        return mask
    sizes = ndimage.sum(mask, labels, range(1, n + 1))
    return labels == (1 + int(np.argmax(sizes)))


def _ball(radius, ndim):
    grid = np.mgrid[tuple(slice(-radius, radius + 1) for _ in range(ndim))]
    return (grid ** 2).sum(0) <= radius ** 2


def heuristic_brain_mask(arr, spacing_mm=1.0, erode_mm=4.0):
    """Rough skull strip without any model (2D or 3D): Otsu threshold, erode to cut
    the brain loose from scalp and skull, keep the largest piece, grow it back and
    fill holes. Good enough to remove most skull and background, NOT as clean as
    HD-BET or SynthStrip. Look at the preview images before trusting it."""
    nz = arr[arr > 0]
    if nz.size == 0:
        return np.zeros(arr.shape, bool)
    head = arr > otsu_threshold(nz)
    r = max(1, int(round(erode_mm / spacing_mm)))
    ball = _ball(r, arr.ndim)
    core = _largest_component(ndimage.binary_erosion(head, ball))
    brain = ndimage.binary_dilation(core, ball) & head
    if arr.ndim == 3:     # fill holes slice by slice too (ventricles are dark on T1)
        brain = ndimage.binary_fill_holes(brain)
        for z in range(brain.shape[2]):
            brain[:, :, z] = ndimage.binary_fill_holes(brain[:, :, z])
    else:
        brain = ndimage.binary_fill_holes(brain)
    return brain


def clean_mask(mask):
    """Largest connected piece with holes filled."""
    if not mask.any():
        return mask
    return ndimage.binary_fill_holes(_largest_component(mask))


# ---------- bias field ----------
def n4_correct(arr, mask, spacing_mm):
    import SimpleITK as sitk   # pip install SimpleITK
    image = sitk.GetImageFromArray(arr.astype(np.float32))
    image.SetSpacing((spacing_mm,) * 3)
    m = sitk.GetImageFromArray(mask.astype(np.uint8))
    m.SetSpacing((spacing_mm,) * 3)
    shrink = 2
    small = sitk.Shrink(image, [shrink] * 3)
    small_mask = sitk.Shrink(m, [shrink] * 3)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([50, 50, 30])
    corrector.Execute(small, small_mask)
    log_bias = corrector.GetLogBiasFieldAsImage(image)
    out = image / sitk.Exp(log_bias)
    return sitk.GetArrayFromImage(out).astype(np.float32) * mask


# ---------- intensity ----------
def fit_normalizer(values, method="zscore"):
    """Fit an intensity mapping on brain voxel values; returns a function values -> normalized.
    Fitting once per volume (or per patient for 2D sources) and applying it to every
    slice keeps relative brightness between slices of the same scan."""
    values = np.asarray(values, np.float32)
    if values.size < 10:
        return lambda v: np.zeros_like(v, np.float32)
    if method == "zscore":
        lo, hi = np.percentile(values, [1, 99])
        core = values[(values >= lo) & (values <= hi)]   # robust to tumor/vessel outliers
        mean, std = float(core.mean()), float(core.std()) + 1e-6
        return lambda v: ((v - mean) / std).astype(np.float32)
    if method == "histmatch":
        landmarks = np.percentile(values, _HM_PERCENTILES)
        landmarks = np.maximum.accumulate(landmarks + np.arange(landmarks.size) * 1e-6)
        # linear extrapolation beyond the outer landmarks
        slope_lo = (_HM_TARGETS[1] - _HM_TARGETS[0]) / (landmarks[1] - landmarks[0] + 1e-6)
        slope_hi = (_HM_TARGETS[-1] - _HM_TARGETS[-2]) / (landmarks[-1] - landmarks[-2] + 1e-6)

        def apply(v):
            mapped = np.interp(v, landmarks, _HM_TARGETS)
            mapped = np.where(v < landmarks[0], _HM_TARGETS[0] + (v - landmarks[0]) * slope_lo, mapped)
            mapped = np.where(v > landmarks[-1], _HM_TARGETS[-1] + (v - landmarks[-1]) * slope_hi, mapped)
            return mapped.astype(np.float32)
        return apply
    raise ValueError(f"Unknown intensity method '{method}'")


def normalize_intensity(arr, brain, method="zscore", normalizer=None):
    """Normalize using brain voxels only. Returns float array, 0 outside the brain."""
    normalizer = normalizer or fit_normalizer(arr[brain], method)
    out = np.zeros(arr.shape, np.float32)
    out[brain] = normalizer(arr[brain])
    return out


def to_gray(norm2d, brain2d, window=Z_WINDOW):
    """Fixed mapping: window[0]..window[1] -> 1..255 inside the brain, 0 outside."""
    lo, hi = window
    g = 1 + 254 * np.clip((norm2d - lo) / (hi - lo), 0, 1)
    return np.where(brain2d, g, 0).astype(np.uint8)


# ---------- the whole volume pipeline ----------
def prepare_volume(image_path, mask_path, cfg: HarmonizeConfig, already_stripped: bool,
                   workdir=None) -> Prepared:
    img = resample_iso(load_canonical(image_path), cfg.spacing_mm, order=1)
    seg = None
    if mask_path:
        seg_img = load_canonical(mask_path)
        seg_img = resample_from_to(seg_img, img, order=0)
        seg = np.rint(np.asarray(seg_img.dataobj)).astype(np.int16)

    stripped = already_stripped
    if not already_stripped and cfg.skullstrip not in ("none", "heuristic"):
        img = run_skullstrip_tool(img, cfg.skullstrip, workdir)
        stripped = True
    arr = np.asarray(img.dataobj, np.float32)
    arr = np.nan_to_num(arr)
    arr[arr < 0] = 0

    if stripped:
        brain = clean_mask(arr > 0)
    elif cfg.skullstrip == "heuristic":
        brain = heuristic_brain_mask(arr, cfg.spacing_mm)
        stripped = True
    else:   # skull stays in: use the whole head, so at least background is removed
        nz = arr[arr > 0]
        brain = clean_mask(arr > otsu_threshold(nz)) if nz.size else arr > 0
    arr = arr * brain

    if cfg.n4:
        arr = n4_correct(arr, brain, cfg.spacing_mm)

    norm = normalize_intensity(arr, brain, cfg.intensity)
    return Prepared(norm=norm, brain=brain, seg=seg, spacing_mm=cfg.spacing_mm,
                    stripped=stripped)


# ---------- slicing ----------
def take_slice(arr, plane, index):
    axis = PLANE_AXIS[plane]
    return np.take(arr, index, axis=axis)


def orient_2d(sl, plane, radiological=True):
    """Display orientation. Axial: anterior up. Coronal/sagittal: superior up."""
    sl = np.rot90(sl)
    if radiological and plane in ("axial", "coronal"):
        sl = np.fliplr(sl)
    return sl


def crop_fov(gray, brain2d, fov_px):
    """Square crop of fov_px pixels centered on the brain's bounding box (zero padded)."""
    rows, cols = np.where(brain2d)
    cy = (rows.min() + rows.max()) // 2 if rows.size else gray.shape[0] // 2
    cx = (cols.min() + cols.max()) // 2 if cols.size else gray.shape[1] // 2
    half = fov_px // 2
    pad = fov_px
    padded = np.pad(gray, pad)
    y0, x0 = cy + pad - half, cx + pad - half
    return padded[y0:y0 + fov_px, x0:x0 + fov_px]


def render_slice(norm2d, brain2d, plane, cfg: HarmonizeConfig, already_oriented=False):
    """Normalized 2D slice at cfg.spacing_mm -> uint8 PIL image of cfg.img_size,
    with the same framing for every source. Slices cut from RAS volumes are
    rotated to display orientation; pass already_oriented=True for sources that
    ship 2D images in display orientation (Figshare .mat slices)."""
    from PIL import Image
    if not already_oriented:
        norm2d = orient_2d(norm2d, plane, cfg.radiological)
        brain2d = orient_2d(brain2d, plane, cfg.radiological)
    gray = to_gray(norm2d, brain2d)
    brain_o = gray > 0
    gray = crop_fov(gray, brain_o, int(round(cfg.fov_mm / cfg.spacing_mm)))
    return Image.fromarray(gray).resize((cfg.img_size, cfg.img_size), Image.BILINEAR)


def brain_range_slices(brain, plane, n, lo=0.35, hi=0.75, min_area_frac=0.5):
    """n evenly spaced slice indices between fractions lo..hi of the brain's extent
    along the plane's axis, keeping only slices with a reasonable amount of brain.
    Used for healthy scans, which have no tumor mask to pick slices with."""
    axis = PLANE_AXIS[plane]
    other = tuple(a for a in range(3) if a != axis)
    area = brain.sum(axis=other)
    idx = np.where(area > 0)[0]
    if idx.size == 0:
        return []
    first, last = idx.min(), idx.max()
    z = np.arange(area.size)
    window = (z >= first + lo * (last - first)) & (z <= first + hi * (last - first))
    ok = np.where(window & (area >= min_area_frac * area.max()))[0]
    if ok.size <= n:
        return list(ok)
    return list(ok[np.linspace(0, ok.size - 1, n).round().astype(int)])


def tumor_slices(seg, brain, plane, n, min_area_frac=0.5):
    """n evenly spaced slices among those with >= 50% of the max tumor area along
    the plane's axis (tumor core when BraTS labels are present), and with a
    reasonable amount of brain so slice levels match the healthy picks."""
    axis = PLANE_AXIS[plane]
    other = tuple(a for a in range(3) if a != axis)
    core = (seg > 0) & (seg != 2)
    mask = core if core.any() else (seg > 0)
    area = mask.sum(axis=other)
    if area.max() == 0:
        return []
    brain_area = brain.sum(axis=other)
    ok = np.where((area >= 0.5 * area.max()) & (brain_area >= min_area_frac * brain_area.max()))[0]
    if ok.size == 0:
        ok = np.where(area >= 0.5 * area.max())[0]
    if ok.size <= n:
        return list(ok)
    return list(ok[np.linspace(0, ok.size - 1, n).round().astype(int)])


def tumor_free_slices(seg, brain, plane, n, margin=10, min_area_frac=0.5):
    """Slices at least `margin` slices away from any tumor label, with enough brain."""
    axis = PLANE_AXIS[plane]
    other = tuple(a for a in range(3) if a != axis)
    has_tumor = (seg > 0).sum(axis=other) > 0
    tz = np.where(has_tumor)[0]
    if tz.size == 0:
        return []
    brain_area = brain.sum(axis=other)
    z = np.arange(seg.shape[axis])
    far = (z < tz.min() - margin) | (z > tz.max() + margin)
    ok = np.where(far & (brain_area >= min_area_frac * brain_area.max()))[0]
    if ok.size <= n:
        return list(ok)
    return list(ok[np.linspace(0, ok.size - 1, n).round().astype(int)])
