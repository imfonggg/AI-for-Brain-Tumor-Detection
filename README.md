# ResNet-50 Brain Tumor Detection Pipeline

## 1. Project Overview & Member Role
- **Model**: ResNet-50 (Deep Residual Learning with Bottleneck Residual Blocks)
- **Assigned Team Member**: Chasrith
- **Project Domain**: Brain Tumor Classification from MRI scans (T1, T1C+, T2)
- **Dataset**: Kaggle Neuro Deep CNN 12K MRI Images / Clinical Brain Tumor Dataset

---

## 2. Model Comparison: ResNet-50 vs. ResNet-18 vs. EfficientNet-B0
This table provides the exact technical talking points for our team meetings and final documentation:

| Specification | Phong's ResNet-18 | Lakshmi's EfficientNet-B0 | Chasrith's ResNet-50 |
| :--- | :--- | :--- | :--- |
| **Residual Block Type** | 2-layer Basic Block ($3\times3 \to 3\times3$) | MBConv (Depthwise Inverted Bottleneck + SE) | **3-layer Bottleneck Block ($1\times1 \to 3\times3 \to 1\times1$)** |
| **Total Layers** | 18 convolutional/FC | 9 stages (MobileNetV3 style) | **50 layers (deep hierarchical features)** |
| **Parameters** | ~11.7 Million | ~5.3 Million | **~25.6 Million (High capacity)** |
| **Clinical Strength** | Fast training on laptops, low memory baseline | Highly parameter-efficient, balanced depth/width | **Captures minute, subtle margins, necrotic cores, and contrast enhancement gradients** |
| **Pretrained Base** | ImageNet-1K | ImageNet-1K | **ImageNet-1K V2** |

### Why ResNet-50's Bottleneck Architecture Matters for MRI:
In ResNet-50, each residual block uses a **bottleneck design**:
1. A $1 \times 1$ convolution reduces channel dimensions (e.g. from 256 down to 64).
2. A $3 \times 3$ convolution performs spatial feature extraction in the lower-dimensional space.
3. A $1 \times 1$ convolution expands the channels back ($4\times$, up to 256).

This allows the network to be much deeper (50 layers) without computational explosion, enabling the network to learn complex tumor textures, heterogenous T1 contrast enhancement patterns, and irregular margins without vanishing gradient issues.

---

## 3. Data Leakage Prevention (Section 2.1)
Medical MRI datasets frequently contain multiple slices from the same patient. Shuffling individual slices results in **severe data leakage**: the model memorizes patient skull geometry and achieves misleadingly high test accuracy that collapses on real patients.

Our pipeline strictly enforces **Patient-Level Stratified Splitting**:
- **80% Training / 10% Validation / 10% Testing**
- Grouping is enforced by `Patient ID`.
- Automated assertion checks ensure:
  $$\text{Patients}_{\text{train}} \cap \text{Patients}_{\text{val}} = \emptyset, \quad \text{Patients}_{\text{train}} \cap \text{Patients}_{\text{test}} = \emptyset$$

---

## 4. Evaluation Suite & Clinical Metrics (Section 2.2)
Aligned with Shraddaya's metrics definitions:
- **Accuracy**: Overall fraction of correct diagnoses.
- **Precision (Positive Predictive Value)**: $\frac{TP}{TP + FP}$ — Minimizes false alarms and unnecessary biopsies.
- **Recall / Sensitivity (True Positive Rate)**: $\frac{TP}{TP + FN}$ — **Core safety metric**: ensures diseased patients are caught.
- **Specificity (True Negative Rate)**: $\frac{TN}{TN + FP}$ — Measures effectiveness at identifying healthy scans.
- **F1-Score**: Harmonic mean balancing precision and recall.
- **Confusion Matrix**: High-resolution heatmap visualizing exact inter-class misclassifications.

---

## 5. Quickstart & Verification

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run End-to-End Pipeline
To run a quick synthetic verification test (proves model compiles, trains, splits without leakage, and plots results):
```bash
python run_pipeline.py --epochs 2 --batch_size 8
```

### Step 3: Run on Real Kaggle Dataset
When the Kaggle dataset is downloaded:
1. Place the class folders into `data/`:
   ```text
   data/
   ├── glioma/
   │   ├── patient001_slice01.png
   │   └── ...
   ├── meningioma/
   ├── pituitary/
   └── no_tumor/
   ```
2. Run full training:
   ```bash
   python run_pipeline.py --epochs 15 --batch_size 16
   ```

### Step 4: Output Files Generated
All evaluation metrics and plots are automatically written to `outputs/`:
- `outputs/checkpoints/resnet50_best.pth`: Saved model weights.
- `outputs/confusion_matrix.png`: Heatmap with raw counts and class-normalized percentages.
- `outputs/training_curves.png`: Side-by-side Train vs. Val Loss and Accuracy plots.
- `outputs/test_metrics.json`: JSON record of all global and per-class clinical metrics.
