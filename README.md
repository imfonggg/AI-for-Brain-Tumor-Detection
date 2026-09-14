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

Baseline CNN (trained from scratch):
```bash
python train.py --model baseline --epochs 15
```

Transfer learning (pretrained ResNet18, recommended — better accuracy, faster to converge):
```bash
python train.py --model resnet18 --epochs 10 --lr 0.0003
```

Try a second architecture for your "compare approaches" requirement:
```bash
python train.py --model efficientnet_b0 --epochs 10 --lr 0.0003
```

Each run saves a checkpoint to `outputs/<model>_best.pt` and a training curve plot to
`outputs/<model>_curves.png`.

## 4. Evaluate

```bash
python evaluate.py --model resnet18 --checkpoint outputs/resnet18_best.pt
```

This prints accuracy, precision, recall, and F1 (overall and per-class), and saves a
confusion matrix image to `outputs/<model>_confusion_matrix.png`.

## 5. Compare models

Run `train.py` + `evaluate.py` for each architecture (baseline, resnet18, efficientnet_b0),
then put the resulting metrics side by side in your report/slides. This directly satisfies
the "compare model performance" and "explore different ML/DL approaches" guidelines.

## Project structure

```
brain_tumor_project/
  data/                 <- put the dataset here (not included)
  data_loader.py         <- dataset class, transforms, train/val split
  model.py                <- baseline CNN + pretrained model factory
  train.py                <- training loop, saves checkpoints + curves
  evaluate.py              <- metrics: accuracy/precision/recall/F1/confusion matrix
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
- This is explicitly a research/prototype tool, not a diagnostic system.
