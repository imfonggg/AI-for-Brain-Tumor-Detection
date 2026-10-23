"""
Data loading utilities for the Brain Tumor MRI classification project.

Expects data laid out as:
    data_real/Training/<class_name>/*.jpg
    data_real/Testing/<class_name>/*.jpg

as written by nifti_slices.py from real 3D MRI scans. Classes come from the
folder names, so any subset of glioma, meningioma, notumor, pituitary works.

Filenames of the form <patientID>_z###.jpg are grouped by patient when
splitting train/val, so no patient appears on both sides. Files without that
pattern are treated as their own group.
"""

import os
import random
import re
from collections import Counter, defaultdict

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
IMG_SIZE = 224
DATA_DIR = "data_real"
PATIENT_PATTERN = re.compile(r"^(.+)_z\d+$")
IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]


def get_transforms(train: bool):
    """Standard ImageNet-style normalization so pretrained backbones work well.
    Augmentation only applied to the training split."""
    if train:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
        ])
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
    ])


def print_class_distribution(dataset, name="dataset"):
    """Check class balance -- medical imaging datasets are often imbalanced,
    and the project guidelines ask you to report this kind of thing."""
    if hasattr(dataset, "targets"):
        labels = dataset.targets
    else:  # a Subset from split_by_patient
        labels = [dataset.dataset.targets[i] for i in dataset.indices]
    counts = Counter(labels)
    idx_to_class = {v: k for k, v in dataset.dataset.class_to_idx.items()} \
        if hasattr(dataset, "dataset") else \
        {v: k for k, v in dataset.class_to_idx.items()}
    print(f"\nClass distribution for {name} (n={len(labels)}):")
    for idx, count in sorted(counts.items()):
        pct = 100 * count / len(labels)
        print(f"  {idx_to_class[idx]:<12} {count:>5} ({pct:.1f}%)")


def patient_id(path):
    """Patient ID from a '<patientID>_z###.jpg' filename, else the filename itself."""
    stem = os.path.splitext(os.path.basename(path))[0]
    match = PATIENT_PATTERN.match(stem)
    return match.group(1) if match else stem


def split_by_patient(samples, val_split, seed=42):
    """Return (train_indices, val_indices) with every patient on one side only."""
    groups = defaultdict(list)
    for index, (path, _) in enumerate(samples):
        groups[patient_id(path)].append(index)
    patients = sorted(groups)
    random.Random(seed).shuffle(patients)

    n_val_target = int(len(samples) * val_split)
    train_indices, val_indices = [], []
    for pid in patients:
        target = val_indices if len(val_indices) < n_val_target else train_indices
        target.extend(groups[pid])
    return train_indices, val_indices


def get_dataloaders(data_dir=DATA_DIR, batch_size=32, val_split=0.15, num_workers=2,
                    augment_train=True):
    """Returns train_loader, val_loader, test_loader, class_names.

    Training/ is split into train/val. Testing/ is used only for final evaluation
    (evaluate.py) so numbers stay honest -- don't peek at it during training.
    """
    train_dir = os.path.join(data_dir, "Training")
    test_dir = os.path.join(data_dir, "Testing")

    if not os.path.isdir(train_dir):
        raise FileNotFoundError(
            f"Couldn't find {train_dir}. Download the dataset and arrange it as "
            f"described in README.md before running this script."
        )

    full_train = datasets.ImageFolder(
        train_dir, transform=get_transforms(train=augment_train)
    )
    class_names = full_train.classes

    train_indices, val_indices = split_by_patient(full_train.samples, val_split)
    train_subset = Subset(full_train, train_indices)
    # val split should NOT use training augmentation -- use eval transforms
    val_subset = Subset(
        datasets.ImageFolder(train_dir, transform=get_transforms(train=False)), val_indices
    )

    test_dataset = datasets.ImageFolder(test_dir, transform=get_transforms(train=False))

    print_class_distribution(train_subset, "train")
    print_class_distribution(val_subset, "validation")

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers)

    return train_loader, val_loader, test_loader, class_names


if __name__ == "__main__":
    # Quick sanity check: python data_loader.py
    train_loader, val_loader, test_loader, class_names = get_dataloaders()
    print(f"\nClasses: {class_names}")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, "
          f"Test batches: {len(test_loader)}")
