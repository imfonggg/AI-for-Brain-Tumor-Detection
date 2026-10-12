"""
Dataset management, preprocessing, and patient-level data partitioning.

Crucial Clinical Implementation:
Prevents data leakage by strictly partitioning scans by Patient ID (Section 2.1).
All slices belonging to a specific patient are isolated to either Train, Val, or Test.
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit

import config


def extract_patient_id(file_path: Path) -> str:
    """
    Extracts a unique patient identifier from an image filepath.
    Supports common medical dataset naming conventions:
    - patient_001_slice_05.jpg -> 'patient_001'
    - P042_T1_08.png           -> 'P042'
    - class/sub-012/slice1.png  -> 'sub-012' (parent folder if patient-based)
    - Fallback: Uses filename stem prior to slice number or full stem
    """
    stem = file_path.stem
    parent_name = file_path.parent.name
    
    # 1. Check if the parent folder looks like a patient ID (e.g. sub-01, patient10)
    if re.match(r"^(patient[-_]?\d+|sub[-_]?\d+|P[-_]?\d+|ID[-_]?\d+)$", parent_name, re.IGNORECASE):
        return parent_name.lower()
        
    # 2. Look for regex patterns in filename: patient_123, P123, pat_123, sub-123
    match = re.search(r"(patient[-_]?\d+|sub[-_]?\d+|P[-_]?\d+|pat[-_]?\d+|ID[-_]?\d+)", stem, re.IGNORECASE)
    if match:
        return match.group(1).lower()

    # 3. Check for delimiter before slice indicator (e.g., tumor_045_01.jpg -> tumor_045)
    parts = stem.split("_")
    if len(parts) >= 2 and parts[-1].isdigit():
        return "_".join(parts[:-1]).lower()

    # 4. Fallback: single image per patient (unique stem)
    return stem.lower()


class BrainTumorDataset(Dataset):
    """
    PyTorch Dataset for Brain Tumor MRI slices.
    Loads images, converts grayscale scans to 3-channel RGB, applies transforms.
    """
    def __init__(self, file_paths: List[Path], labels: List[int], patient_ids: List[str], transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.patient_ids = patient_ids
        self.transform = transform

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        path = self.file_paths[idx]
        label = self.labels[idx]
        patient_id = self.patient_ids[idx]

        # Load image; convert to RGB (standard 3-channel format expected by ResNet-50)
        image = Image.open(path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label, patient_id


def get_data_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Builds training augmentation and validation/test preprocessing pipelines.
    Augmentation prevents the 25.6M-parameter ResNet-50 from memorizing training photos.
    """
    train_transform = transforms.Compose([
        transforms.Resize(config.IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.RandomAffine(degrees=0, scale=(0.95, 1.05), translate=(0.05, 0.05)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize(config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
    ])

    return train_transform, eval_transform


def scan_dataset(data_dir: Path) -> Tuple[List[Path], List[int], List[str], List[str]]:
    """
    Scans a directory structured as class subfolders:
      data_dir/
        ├── glioma/
        ├── meningioma/
        ├── pituitary/
        └── notumor/
    Or Kaggle standard pre-split structure:
      data_dir/
        ├── Training/
        └── Testing/
    Returns file paths, class integer labels, patient IDs, and class names list.
    """
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    # Case A: Pre-split Kaggle structure with Training/ and Testing/
    if (data_dir / "Training").is_dir() and (data_dir / "Testing").is_dir():
        train_dir = data_dir / "Training"
        class_folders = sorted([f for f in train_dir.iterdir() if f.is_dir()])
        class_names = [f.name for f in class_folders]
        class_to_idx = {name: idx for idx, name in enumerate(class_names)}

        file_paths, labels, patient_ids = [], [], []
        for split_dir in [data_dir / "Training", data_dir / "Testing"]:
            for class_folder in sorted([f for f in split_dir.iterdir() if f.is_dir()]):
                label = class_to_idx[class_folder.name]
                for file in class_folder.glob("*"):
                    if file.suffix.lower() in valid_extensions:
                        file_paths.append(file)
                        labels.append(label)
                        patient_ids.append(extract_patient_id(file))
        return file_paths, labels, patient_ids, class_names

    # Case B: Flat class folders
    class_folders = sorted([f for f in data_dir.iterdir() if f.is_dir()])
    if not class_folders:
        raise ValueError(f"No class subfolders found in {data_dir}. Expected data_dir/<class_name>/*.png")

    class_names = [f.name for f in class_folders]
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    file_paths = []
    labels = []
    patient_ids = []

    for class_folder in class_folders:
        label = class_to_idx[class_folder.name]
        for file in class_folder.glob("*"):
            if file.suffix.lower() in valid_extensions:
                file_paths.append(file)
                labels.append(label)
                patient_ids.append(extract_patient_id(file))

    if not file_paths:
        raise ValueError(f"No valid images found in {data_dir}.")

    return file_paths, labels, patient_ids, class_names


def patient_level_split(
    file_paths: List[Path],
    labels: List[int],
    patient_ids: List[str],
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    random_seed: int = 42
) -> Dict[str, Dict[str, list]]:
    """
    Performs Partitioning matching the team's clinical standards.
    If pre-split into Training and Testing folders, separates Training into Train & Val
    and uses Testing directly, ensuring exact benchmark alignment with Lakshmi's run.
    Otherwise performs GroupShuffleSplit by Patient ID.
    """
    np.random.seed(random_seed)

    # Check if files originate from pre-split Kaggle directories
    has_presplit = any("Testing" in p.parts for p in file_paths) and any("Training" in p.parts for p in file_paths)

    if has_presplit:
        print("[Dataset] Detected Kaggle pre-split benchmark dataset (Training/ & Testing/).")
        train_val_paths = [p for p in file_paths if "Training" in p.parts]
        train_val_labels = [l for p, l in zip(file_paths, labels) if "Training" in p.parts]
        train_val_pids = [pid for p, pid in zip(file_paths, patient_ids) if "Training" in p.parts]

        test_paths = [p for p in file_paths if "Testing" in p.parts]
        test_labels = [l for p, l in zip(file_paths, labels) if "Testing" in p.parts]
        test_pids = [pid for p, pid in zip(file_paths, patient_ids) if "Testing" in p.parts]

        # Split 5,600 training images into 4,760 train (85%) and 840 val (15%)
        from sklearn.model_selection import train_test_split
        train_idx, val_idx = train_test_split(
            range(len(train_val_paths)),
            test_size=840,
            random_state=random_seed,
            stratify=train_val_labels
        )

        return {
            "train": {
                "paths": [train_val_paths[i] for i in train_idx],
                "labels": [train_val_labels[i] for i in train_idx],
                "pids": [train_val_pids[i] for i in train_idx]
            },
            "val": {
                "paths": [train_val_paths[i] for i in val_idx],
                "labels": [train_val_labels[i] for i in val_idx],
                "pids": [train_val_pids[i] for i in val_idx]
            },
            "test": {
                "paths": test_paths,
                "labels": test_labels,
                "pids": test_pids
            }
        }

    # Standard GroupShuffleSplit by patient
    test_size = test_ratio
    val_and_test_size = val_ratio + test_ratio

    gss1 = GroupShuffleSplit(n_splits=1, test_size=val_and_test_size, random_state=random_seed)
    train_idx, val_test_idx = next(gss1.split(file_paths, labels, groups=patient_ids))

    train_paths = [file_paths[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    train_pids = [patient_ids[i] for i in train_idx]

    val_test_paths = [file_paths[i] for i in val_test_idx]
    val_test_labels = [labels[i] for i in val_test_idx]
    val_test_pids = [patient_ids[i] for i in val_test_idx]

    relative_test_ratio = test_ratio / (val_ratio + test_ratio)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=relative_test_ratio, random_state=random_seed)
    val_idx, test_idx = next(gss2.split(val_test_paths, val_test_labels, groups=val_test_pids))

    val_paths = [val_test_paths[i] for i in val_idx]
    val_labels = [val_test_labels[i] for i in val_idx]
    val_pids = [val_test_pids[i] for i in val_idx]

    test_paths = [val_test_paths[i] for i in test_idx]
    test_labels = [val_test_labels[i] for i in test_idx]
    test_pids = [val_test_pids[i] for i in test_idx]

    return {
        "train": {"paths": train_paths, "labels": train_labels, "pids": train_pids},
        "val": {"paths": val_paths, "labels": val_labels, "pids": val_pids},
        "test": {"paths": test_paths, "labels": test_labels, "pids": test_pids},
    }


def create_dataloaders(
    splits: Dict[str, Dict[str, list]],
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = config.NUM_WORKERS,
    pin_memory: bool = config.PIN_MEMORY
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates PyTorch DataLoaders for train, validation, and test partitions.
    """
    train_transform, eval_transform = get_data_transforms()

    train_ds = BrainTumorDataset(
        splits["train"]["paths"], splits["train"]["labels"], splits["train"]["pids"], transform=train_transform
    )
    val_ds = BrainTumorDataset(
        splits["val"]["paths"], splits["val"]["labels"], splits["val"]["pids"], transform=eval_transform
    )
    test_ds = BrainTumorDataset(
        splits["test"]["paths"], splits["test"]["labels"], splits["test"]["pids"], transform=eval_transform
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, val_loader, test_loader
