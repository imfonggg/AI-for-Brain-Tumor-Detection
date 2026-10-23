# AI for Brain Tumor Detection

Classify brain MRI slices by tumor type. The model trains on 2D slices cut from
real 3D MRI scans (NIfTI files). Current classes: **glioma, meningioma, no tumor**
(pituitary and other types will be added as 3D sources are found).

This is a research/prototype project — NOT a clinical diagnostic tool.

## 1. Build the dataset from NIfTI scans

`nifti_slices.py` turns 3D volumes into axial JPEG slices:

| Class      | Source                                              | Sequence           |
|------------|-----------------------------------------------------|--------------------|
| glioma     | BraTS 2021 (UCSF-PDGM cases by default)             | T1 post-contrast   |
| meningioma | BraTS-style meningioma set                          | T1 post-contrast   |
| notumor    | IXI healthy controls (streamed from the IXI server) | T1                 |

Tumor slices are picked where the segmentation mask shows the most tumor; IXI
slices come from the mid brain. Every patient goes entirely into `Training/` or
`Testing/` (20% of patients held out), and `data_loader.py` also splits train/val
by patient, so no patient's slices appear on both sides of any split.

The easiest way to run it is the Kaggle notebook `nfti-to-jpeg-conversion.ipynb`
(attach the BraTS and meningioma datasets, turn Internet on, run all). It writes
`data_real.zip`. Or run the script directly:

```bash
python nifti_slices.py --out_dir data_real \
    --glioma_root /path/to/brats-2021 \
    --meningioma_root /path/to/meningioma \
    --ixi
```

Output layout (the default `--data_dir` for training and evaluation):

```
data_real/
  Training/{glioma,meningioma,notumor}/<patientID>_z###.jpg
  Testing/{glioma,meningioma,notumor}/<patientID>_z###.jpg
```

The older Kaggle JPEG dataset in `data/` is no longer used by default; pass
`--data_dir data` to the scripts to run on it.

## 2. Install dependencies

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install nibabel requests pandas   # only needed for nifti_slices.py
```

## 3. Train

Train the requested ResNet50 in two phases. The default is 5 classifier-head epochs
followed by 10 fine-tuning epochs:

```bash
python train.py
```

To choose the number of epochs for each phase:

```bash
python train.py --head_epochs 5 --epochs 8
```

The head-only phase uses Adam at `1e-3`; fine-tuning unfreezes the full network
and uses Adam at `1e-5`. Each epoch prints train/validation loss and accuracy,
and the checkpoint with the best validation accuracy is kept.

Training saves `outputs/resnet50_best.pt` and `outputs/resnet50_curves.png`.

## 3b. Train EfficientNet-B0 instead

Both `train.py` and `evaluate.py` now take a `--model` flag
(`resnet50` or `efficientnet_b0`; defaults to `resnet50` so existing
commands still work unchanged):

```
python train.py --model efficientnet_b0 --head_epochs 5 --epochs 8
```

This runs the same two-phase schedule as ResNet50 (freeze the pretrained
backbone and train just the classifier head first, then unfreeze and
fine-tune the whole network at a lower learning rate). It saves to
`outputs/efficientnet_b0_best.pt` and `outputs/efficientnet_b0_curves.png`
so it won't overwrite a ResNet50 run.

Then evaluate the same way as before — the checkpoint remembers which
architecture it is, so `--model` doesn't need to be passed again:

```
python evaluate.py --checkpoint outputs/efficientnet_b0_best.pt
```

This saves `outputs/efficientnet_b0_confusion_matrix.png`.

**Note:** `hybrid_svm.py` and `grad_cam.py` still assume a ResNet50
checkpoint (they reference `model.fc` and `model.layer4[-1]` directly) and
have not been updated to support EfficientNet-B0 in this branch.

## 3c. Train both models on Google Colab

`train_colab.ipynb` trains ResNet50 and EfficientNet-B0 one after the other on a
Colab T4 GPU. Put `data_real.zip` in `MyDrive/brain_tumor/`, open the notebook in
Colab, pick the T4 runtime and run all cells. Results are saved to
`MyDrive/brain_tumor/outputs/`; re-running after a disconnect skips any model that
already finished.

To compare models yourself, evaluate each one with `--output_csv`, then:

```bash
python compare_models.py outputs/resnet50_metrics.csv outputs/efficientnet_b0_metrics.csv
```

This prints overall and per-class results side by side and saves
`outputs/model_comparison.csv` and `outputs/model_comparison.png`.

## 4. Evaluate

```bash
python evaluate.py --checkpoint outputs/resnet50_best.pt
```

This prints accuracy, precision, recall, and F1 (overall and per-class), and saves a
confusion matrix image to `outputs/resnet50_confusion_matrix.png`. Check per-class
recall for glioma, meningioma, and pituitary rather than relying on accuracy alone.

To also save the overall and per-class metrics to a CSV file, provide its path with
`--output_csv`:

```bash
python evaluate.py --checkpoint outputs/resnet50_best.pt --output_csv outputs/resnet50_metrics.csv
```

Choose a different path or filename to keep separate evaluation results.

## 5. Compare the softmax and SVM heads

Run the hybrid comparison using the trained ResNet50 checkpoint:

```bash
python hybrid_svm.py --checkpoint outputs/resnet50_best.pt
```

It extracts 2048-dimensional ResNet50 features, fits `StandardScaler` + an RBF SVM
on the training split, and reports both classifiers on the same held-out `Testing/`
images. It saves both confusion matrices and `outputs/resnet50_svm.joblib`.

## 6. Inspect a Grad-CAM heatmap

```bash
python grad_cam.py --checkpoint outputs/resnet50_best.pt --image data_real/Testing/glioma/UCSF-PDGM-004_z071.jpg
```

The overlay is saved to `outputs/gradcam.png`. Grad-CAM is a qualitative inspection
tool, not evidence that the model is clinically reliable or attending to the correct
medical features.

## Project structure

```
brain_tumor_project/
  nifti_slices.py         <- converts NIfTI volumes into patient-split JPEG slices
  nfti-to-jpeg-conversion.ipynb <- Kaggle notebook that runs nifti_slices.py
  data_real/            <- generated slices (not included)
  data_loader.py         <- dataset class, transforms, train/val split
  model.py                <- pretrained ResNet50 construction
  train.py                <- training loop, saves checkpoints + curves
  train_colab.ipynb       <- trains and compares both models on a Colab GPU
  compare_models.py       <- side-by-side results for several models
  evaluate.py              <- metrics: accuracy/precision/recall/F1/confusion matrix
  hybrid_svm.py            <- compares softmax and ResNet50-feature SVM classifiers
  grad_cam.py              <- creates a Grad-CAM image overlay
  requirements.txt
  outputs/                <- checkpoints, plots, results land here
```

## Notes on limitations (for your write-up)

- Public dataset of unknown provenance/scanner variability — real clinical deployment
  would need multi-site data and radiologist-verified labels.
- No external validation set from a different institution/scanner.
- Class imbalance should be checked and reported (see `data_loader.py`'s
  `print_class_distribution` helper).
- 2D slice classification ignores 3D spatial context that a radiologist would use.
- Tumor classes are T1 post-contrast from BraTS scanners while healthy controls are
  plain T1 from IXI hospitals, so a model could learn contrast/scanner differences
  instead of tumor features. `--notumor_from_tumor_patients` in `nifti_slices.py`
  adds same-scanner notumor slices to reduce this; compare results with and without it.
- This is explicitly a research/prototype tool, not a diagnostic system.
