"""
Data loading utilities for the Brain Tumor MRI classification project.

Expects data laid out as:
    data/Training/<class_name>/*.jpg
    data/Testing/<class_name>/*.jpg

where <class_name> is one of: glioma, meningioma, pituitary, notumor
(folder names from the Kaggle "Brain Tumor MRI Dataset" -- adjust
CLASS_NAMES below if your download uses different folder names).
"""

import os
from collections import Counter

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
IMG_SIZE = 224
DATA_DIR = "data"


def get_transforms(train: bool):
    """Standard ImageNet-style normalization so pretrained backbones work well.
    Augmentation only applied to the training split."""
    if train:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225]),
    ])


def print_class_distribution(dataset, name="dataset"):
    """Check class balance -- medical imaging datasets are often imbalanced,
    and the project guidelines ask you to report this kind of thing."""
    if hasattr(dataset, "targets"):
        labels = dataset.targets
    else:  # a Subset from random_split
        labels = [dataset.dataset.targets[i] for i in dataset.indices]
    counts = Counter(labels)
    idx_to_class = {v: k for k, v in dataset.dataset.class_to_idx.items()} \
        if hasattr(dataset, "dataset") else \
        {v: k for k, v in dataset.class_to_idx.items()}
    print(f"\nClass distribution for {name} (n={len(labels)}):")
    for idx, count in sorted(counts.items()):
        pct = 100 * count / len(labels)
        print(f"  {idx_to_class[idx]:<12} {count:>5} ({pct:.1f}%)")


def get_dataloaders(data_dir=DATA_DIR, batch_size=32, val_split=0.15, num_workers=2):
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

    full_train = datasets.ImageFolder(train_dir, transform=get_transforms(train=True))
    class_names = full_train.classes

    n_val = int(len(full_train) * val_split)
    n_train = len(full_train) - n_val
    train_subset, val_subset = random_split(
        full_train, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    # val split should NOT use training augmentation -- swap in eval transforms
    val_subset.dataset = datasets.ImageFolder(train_dir, transform=get_transforms(train=False))

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
