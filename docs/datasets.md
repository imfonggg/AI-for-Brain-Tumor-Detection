# 3D MRI sources for more tumor classes

Candidate NIfTI/DICOM datasets to grow the real-scan training set beyond glioma,
meningioma and no tumor. Researched October 2026; check each license before use.

## Pituitary

| Dataset | Patients | Scans | Masks | Access | Fit |
|---|---|---|---|---|---|
| [Pituitary NET MRI dataset](https://www.nature.com/articles/s41597-024-04218-8) (Figshare `10.6084/m9.figshare.27894084`) | 136 | Whole-brain T1 post-contrast (+T2 for most), NIfTI, resampled to 0.5 mm | Tumour + gland mask, carotid mask | Open download; article CC BY 4.0, check the Figshare record's own license | **Best fit.** Supported now via `nifti_slices.py --pituitary_root` |
| [Pituitary adenoma multimodal MRI](https://github.com/DrMartinCerny/pituitary_adenoma_multimodal_MRI_dataset) ([OpenNeuro ds006248](https://openneuro.org/datasets/ds006248)) | 50 | Up to 10 sequences, BIDS NIfTI, includes coronal T1 post-contrast | Tumour, gland, carotids, optic chiasm | Open, CC BY-NC 4.0 | Good second source. Coronal acquisition, so slices need to be resliced to axial or taken as coronal |
| [Pituitary Adenoma MRI Segmentation](https://ieee-dataport.org/documents/pituitary-adenoma-mri-segmentation-dataset) (IEEE DataPort) | 55 | 3D T1 post-contrast | Binary tumour mask | Needs an IEEE DataPort subscription | Only if you already have access |

Together the first two give about 186 pituitary patients, below the 400 per class
cap used for the other classes. Lower `--max_patients` for the other classes or
weight the loss if pituitary ends up much smaller.

Pituitary tumors are small and sit at the skull base, so `pick_tumor_slices` will
choose low axial slices. Check the preview grid in the Kaggle notebook.

## Other tumor types

| Type | Dataset | Patients | Scans | Access |
|---|---|---|---|---|
| Brain metastases | [BraTS-METS 2025](https://arxiv.org/pdf/2504.12527) | ~1,300 training | T1, T1 post-contrast, FLAIR; NIfTI, skull-stripped, BraTS layout | Synapse account + data use terms ([page](https://www.synapse.org/Synapse:syn64153130/wiki/631048)) |
| Pediatric glioma | [BraTS-PEDs](https://arxiv.org/pdf/2404.15009) | a few hundred | BraTS layout | Synapse, same process as BraTS-METS |
| Vestibular schwannoma | [Vestibular-Schwannoma-SEG](https://portal.imaging.datacommons.cancer.gov/collections/vestibular_schwannoma_seg) (TCIA/IDC) | 242 | T1 post-contrast + T2, DICOM with RTSTRUCT contours | `pip install idc-index && idc download vestibular_schwannoma_seg`; needs DICOM-to-NIfTI conversion first |

BraTS-METS and BraTS-PEDs use the same folder layout as BraTS 2021, so
`process_brats` reads them with no code change (point a new `--*_root` flag at the
folder). Pediatric glioma is still glioma, so it would add data to that class
rather than creating a new one.

## Watch out for scan-type shortcuts

Each class coming from a different dataset means each class also has its own
scanner, sequence and preprocessing. BraTS-style sets are skull-stripped and
resampled to 1 mm. IXI and the pituitary sets keep the skull and use other
resolutions. A model can reach high accuracy by recognising those differences
instead of the tumor. Ways to reduce this:

- Skull-strip every non-BraTS volume (for example with SynthStrip or HD-BET)
  before slicing, so all classes look alike outside the brain.
- Use `--notumor_from_tumor_patients` so some notumor slices come from the same
  scanners as the tumor classes.
- Look at Grad-CAM on test slices: heatmaps on the skull or background mean
  the model has found a shortcut.
