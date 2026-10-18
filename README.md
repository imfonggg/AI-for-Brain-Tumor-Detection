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

**Using a different dataset or folder:** pass its location with `--data_dir`, for example
`python train.py --data_dir /content/my_dataset`. Use the same `--data_dir` for
`train.py` and `evaluate.py`.

**How the data is split.** The loader looks at what the folder contains:

| Folder contents                        | Split used                                                                      |
| -------------------------------------- | ------------------------------------------------------------------------------- |
| `Training/`, `Validation/`, `Testing/` | Used exactly as provided                                                        |
| `Training/` and `Testing/` only        | 15% of `Training/` becomes validation                                           |
| One folder per class, no split         | Split automatically: 80% train / 10% validation / 10% test, stratified by class |
| Any of the above with `--resplit`      | All images pooled and re-split 80/10/10, stratified by class                    |

Splits use a fixed seed, and the files on disk are never modified. A class needs at least
3 images to be split.

`--resplit` is for getting a true 80/10/10 split from a dataset that came split some other
way, for example `python train.py --data_dir /content/my_dataset --resplit`. Do **not** use it
on a dataset that was split by patient, because it would mix a patient's images across
train, validation, and test. `evaluate.py` and `hybrid_svm.py` read the setting from the
checkpoint, so you don't pass `--resplit` again.

Datasets without patient IDs can have near-duplicate or same-patient images land in
different splits, so treat their results as optimistic.

## 2. Install dependencies

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 3. Train

Train the default ResNet50 in two phases. The default is 5 classifier-head epochs
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

## 3b. Train a different model

`train.py` takes a `--model` flag with three choices: `resnet18`, `resnet50`, and
`efficientnet_b0`. It defaults to `resnet50`, so existing commands still work unchanged.

```bash
python train.py --model resnet18 --head_epochs 5 --epochs 8
python train.py --model efficientnet_b0 --head_epochs 5 --epochs 8
```

Every model uses the same two-phase schedule as ResNet50 (freeze the pretrained
backbone and train just the classifier head first, then unfreeze and fine-tune the
whole network at a lower learning rate). Each model saves to its own files,
`outputs/{model}_best.pt` and `outputs/{model}_curves.png`, so training a different
model won't overwrite another model's checkpoint.

Training the same model a second time overwrites its checkpoint. Rename or move
`outputs/{model}_best.pt` first if you want to keep it.

## 4. Evaluate

```bash
python evaluate.py --checkpoint outputs/resnet50_best.pt
python evaluate.py --checkpoint outputs/resnet18_best.pt
python evaluate.py --checkpoint outputs/efficientnet_b0_best.pt
```

The checkpoint remembers which architecture it is, so `--model` doesn't need to be passed.

Results are saved to a CSV file instead of being printed. Each run of each model gets
its own file, named `outputs/{model}_{run}_results.csv`. The run label is numbered
automatically (`run1`, `run2`, ... separately for each model). To name a run yourself,
use `--run_label`:

```bash
python evaluate.py --checkpoint outputs/resnet18_best.pt --run_label lr-test
```

An existing results file is never overwritten; the script stops with an error if the
file for that model and run label already exists. A confusion matrix image is saved
alongside it as `outputs/{model}_{run}_confusion_matrix.png`.

### Results CSV layout

Each results file has the same 11 columns and 7 rows:

| Rows                                                   | What they hold                                                                                    |
| ------------------------------------------------------ | ------------------------------------------------------------------------------------------------- |
| One per class (glioma, meningioma, notumor, pituitary) | train and validation image counts and percents; test precision, recall, F1, and support           |
| `macro avg`, `weighted avg`                            | test precision, recall, F1, and support                                                           |
| `overall`                                              | total train and validation images, total test images, best validation accuracy, and test accuracy |

Columns: `class, train_count, train_percent, val_count, val_percent, test_precision,
test_recall, test_f1, test_support, best_val_accuracy, test_accuracy`.

Check per-class recall for glioma, meningioma, and pituitary rather than relying on
accuracy alone.

## 5. Compare the softmax and SVM heads

Run the hybrid comparison using the trained ResNet50 checkpoint:

```bash
python hybrid_svm.py --checkpoint outputs/resnet50_best.pt
```

It extracts 2048-dimensional ResNet50 features, fits `StandardScaler` + an RBF SVM
on the training split, and reports both classifiers on the same held-out `Testing/`
images. It saves both confusion matrices and `outputs/resnet50_svm.joblib`.

**Note:** this script still assumes a ResNet50 checkpoint. It has not been updated for
ResNet18 (512-dimensional features) or EfficientNet-B0.

## 6. Inspect a Grad-CAM heatmap

```bash
python grad_cam.py --checkpoint outputs/resnet50_best.pt --image data/Testing/glioma/example.jpg
```

The overlay is saved to `outputs/gradcam.png`. Grad-CAM is a qualitative inspection
tool, not evidence that the model is clinically reliable or attending to the correct
medical features.

**Note:** this script still assumes a ResNet50 checkpoint (it references `model.fc` and
`model.layer4[-1]` directly) and has not been updated for the other models.

## Project structure

```
brain_tumor_project/
  data/                 <- put the dataset here (not included)
  data_loader.py         <- dataset class, transforms, train/val split
  model.py                <- pretrained ResNet18, ResNet50, and EfficientNet-B0 construction
  train.py                <- training loop, saves checkpoints + curves
  evaluate.py              <- test-set metrics; writes the per-run results CSV + confusion matrix
  results_csv.py           <- builds the results CSV, auto-numbers runs
  hybrid_svm.py            <- compares softmax and ResNet50-feature SVM classifiers
  grad_cam.py              <- creates a Grad-CAM image overlay
  requirements.txt
  outputs/                <- checkpoints, plots, results CSVs land here
```

## Notes on limitations (for your write-up)

- Public dataset of unknown provenance/scanner variability — real clinical deployment
  would need multi-site data and radiologist-verified labels.
- No external validation set from a different institution/scanner.
- Class imbalance should be checked and reported (see `data_loader.py`'s
  `print_class_distribution` helper; the results CSV also records train and
  validation class counts).
- 2D slice classification ignores 3D spatial context that a radiologist would use.
- There are no patient IDs, and near-duplicate images may occur across the provided
  training and testing folders. Reported test performance may therefore be optimistic.
- This is explicitly a research/prototype tool, not a diagnostic system.
