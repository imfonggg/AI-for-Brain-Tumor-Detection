"""
Configuration module for ResNet-50 Brain Tumor Detection.
Centralizes all hyperparameters, directory paths, and clinical split settings.
"""

import os
from pathlib import Path
import torch

# Base directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"

# Ensure output directories exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# File paths
BEST_MODEL_PATH = CHECKPOINT_DIR / "resnet50_best.pth"
CONFUSION_MATRIX_PATH = OUTPUT_DIR / "confusion_matrix.png"
TRAINING_CURVES_PATH = OUTPUT_DIR / "training_curves.png"
METRICS_JSON_PATH = OUTPUT_DIR / "test_metrics.json"

# Hardware configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cpu":
    torch.set_num_threads(10)
NUM_WORKERS = 0  # 0 is safe across all platforms (especially Windows)
PIN_MEMORY = torch.cuda.is_available()

# Reproducibility
RANDOM_SEED = 42

# Image Preprocessing & Model Input (Section 3.2 of Master Doc)
IMAGE_SIZE = (224, 224)
IN_CHANNELS = 3

# ImageNet normalization standard (used for transfer learning)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Data Partitioning Settings (Section 2.1 of Master Doc)
# Recommended ratio: 80% Train / 10% Validation / 10% Testing
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

# Hyperparameters
BATCH_SIZE = 32  # Optimized for 10 CPU threads
NUM_EPOCHS = 3
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 3

# Transfer Learning Strategy
# Phase 1: Feature Extraction (head only), Phase 2: Fine-tuning (deep layers)
FINE_TUNE = True
FINE_TUNE_START_EPOCH = 3
FINE_TUNE_LR = 1e-5
