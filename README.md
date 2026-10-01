# AI for Brain Tumor Detection

Classify brain MRI scans into 4 classes: **glioma, meningioma, pituitary, no tumor**.

This is a research/prototype project — NOT a clinical diagnostic tool.

## 1. Get the dataset

Recommended: Kaggle "Brain Tumor MRI Dataset" (4-class, pre-split into folders).

1. Go to Kaggle and search "Brain Tumor MRI Dataset" (Masoud Nickparvar's version is the
   most widely used — ~7,000 images, glioma/meningioma/pituitary/notumor).
2. Download and unzip it.
3. Arrange it like this inside `data/`:

```
data/
  Training/
    glioma/
    meningioma/
    pituitary/
    notumor/
  Testing/
    glioma/
    meningioma/
    pituitary/
    notumor/
```

If your download already has this `Training/Testing` + class-folder structure, just drop
it straight into `data/`.

## 2. Install dependencies

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
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

## 4. Evaluate

```bash
python evaluate.py --checkpoint outputs/resnet50_best.pt
```

This prints accuracy, precision, recall, and F1 (overall and per-class), and saves a
confusion matrix image to `outputs/resnet50_confusion_matrix.png`. Check per-class
recall for glioma, meningioma, and pituitary rather than relying on accuracy alone.

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
python grad_cam.py --checkpoint outputs/resnet50_best.pt --image data/Testing/glioma/example.jpg
```
The overlay is saved to `outputs/gradcam.png`. Grad-CAM is a qualitative inspection
tool, not evidence that the model is clinically reliable or attending to the correct
medical features.

## Project structure

```
brain_tumor_project/
  data/                 <- put the dataset here (not included)
  data_loader.py         <- dataset class, transforms, train/val split
  model.py                <- pretrained ResNet50 construction
  train.py                <- training loop, saves checkpoints + curves
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
- There are no patient IDs, and near-duplicate images may occur across the provided
  training and testing folders. Reported test performance may therefore be optimistic.
- This is explicitly a research/prototype tool, not a diagnostic system.
