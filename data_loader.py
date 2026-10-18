"""
Data loading utilities for the Brain Tumor MRI classification project.

Expects a data folder laid out as:
    <data_dir>/Training/<class_name>/*.jpg
    <data_dir>/Testing/<class_name>/*.jpg

<data_dir> can be any path, e.g. "data" in the repo or "/content/my_dataset"
in Google Colab (pass it with --data_dir). The split folders are found
automatically, ignoring case, and "train"/"test" work as well as
"Training"/"Testing". If the dataset sits inside one or two wrapper folders
(common after unzipping a download), those are skipped automatically. Class
names are taken from the class folder names, so datasets with different
classes work too.

How the data is split depends on what the folder contains:
  * Training/ + Validation/ + Testing/ : used exactly as provided.
  * Training/ + Testing/ only          : 15% of Training/ becomes validation.
  * one folder per class, no split     : split automatically 80/10/10, stratified.
  * resplit=True (--resplit)           : pools every image from the split folders
                                         and re-splits 80/10/10, stratified.
The files on disk are never modified, and training and evaluation get the same
split as long as they use the same data folder (and the same resplit setting).
"""

import os
import random
from collections import Counter

import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets, transforms
from torchvision.datasets.folder import default_loader

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
IMG_SIZE = 224
DATA_DIR = "data"


TRAIN_NAMES = ("training", "train")
TEST_NAMES = ("testing", "test")


def _subdirs(path):
    """Subfolders of path, ignoring hidden folders and zip leftovers like __MACOSX."""
    return [d for d in sorted(os.listdir(path))
            if os.path.isdir(os.path.join(path, d)) and not d.startswith((".", "__"))]


def _find_dir(path, names):
    """The subfolder of path whose lowercase name is in names, or None."""
    for d in _subdirs(path):
        if d.lower() in names:
            return os.path.join(path, d)
    return None


def resolve_split_dirs(data_dir):
    """Locate the training and testing folders inside data_dir.

    Accepts Training/Testing or train/test (any capitalization), and looks
    through up to two single wrapper folders, as left by unzipping a download.
    """
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"Data folder '{data_dir}' does not exist. In Colab, right-click the "
            f"folder in the Files panel and choose 'Copy path'."
        )
    root = data_dir
    for _ in range(3):
        train_dir, test_dir = _find_dir(root, TRAIN_NAMES), _find_dir(root, TEST_NAMES)
        if train_dir and test_dir:
            return train_dir, test_dir
        subfolders = _subdirs(root)
        if len(subfolders) != 1:
            break
        root = os.path.join(root, subfolders[0])
    raise FileNotFoundError(
        f"Couldn't find both a training and a testing folder in '{data_dir}'. "
        f"Expected subfolders named Training/Testing (or train/test), but found: "
        f"{_subdirs(data_dir) or 'no subfolders'}."
    )


VAL_NAMES = ("validation", "valid", "val")
NON_CLASS_NAMES = set(TRAIN_NAMES + TEST_NAMES + VAL_NAMES)


def merge_split_samples(split_samples, split_classes):
    """Pool (path, label) lists from several split folders into one list.

    Every folder must have the same class names. The pooled list is sorted by
    (label, path) so the later split does not depend on which folder an image
    started in.
    """
    reference = list(split_classes[0])
    for classes in split_classes[1:]:
        if list(classes) != reference:
            raise ValueError(
                f"Class folders differ between splits: {reference} vs {list(classes)}."
            )
    pooled = [item for samples in split_samples for item in samples]
    pooled.sort(key=lambda item: (item[1], item[0]))
    return pooled, reference


def find_class_root(data_dir):
    """If data_dir (or a single wrapper folder inside it) holds one folder per
    class, return that folder. Returns None for anything else, including
    partially split datasets (e.g. train/ and valid/ but no test/)."""
    if not os.path.isdir(data_dir):
        return None
    root = data_dir
    for _ in range(3):
        subfolders = _subdirs(root)
        if len(subfolders) >= 2:
            if any(d.lower() in NON_CLASS_NAMES for d in subfolders):
                return None
            return root
        if len(subfolders) == 1:
            root = os.path.join(root, subfolders[0])
        else:
            return None
    return None


def stratified_split_indices(targets, class_names, val_frac, test_frac, seed=42):
    """Split sample indices into train/val/test, keeping each class's share the
    same in every split. Deterministic for a given seed and file ordering."""
    by_class = {}
    for index, label in enumerate(targets):
        by_class.setdefault(label, []).append(index)

    rng = random.Random(seed)
    train_idx, val_idx, test_idx = [], [], []
    for label in sorted(by_class):
        indices = by_class[label]
        if len(indices) < 3:
            raise ValueError(
                f"Class '{class_names[label]}' has only {len(indices)} image(s); "
                f"need at least 3 to split into train/validation/test."
            )
        rng.shuffle(indices)
        n_test = max(1, round(len(indices) * test_frac))
        n_val = max(1, round(len(indices) * val_frac))
        while n_test + n_val >= len(indices):  # always leave at least 1 for training
            if n_test >= n_val and n_test > 1:
                n_test -= 1
            elif n_val > 1:
                n_val -= 1
            else:
                break
        test_idx += indices[:n_test]
        val_idx += indices[n_test:n_test + n_val]
        train_idx += indices[n_test + n_val:]
    return train_idx, val_idx, test_idx


def get_transforms(train: bool):
    """Standard ImageNet-style normalization so pretrained backbones work well.
    Augmentation only applied to the training split."""
    if train:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
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


class PooledImageDataset(Dataset):
    """Images pooled from several folders so they can be re-split together."""

    def __init__(self, samples, classes, transform):
        self.samples = samples
        self.classes = list(classes)
        self.class_to_idx = {name: i for i, name in enumerate(self.classes)}
        self.targets = [label for _, label in samples]
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        return self.transform(default_loader(path)), label


def _split_into_loaders(train_ds, eval_ds, class_names, batch_size, num_workers,
                        verbose, val_frac, test_frac, reason):
    """Stratified 80/10/10-style split of one pool of images into three loaders."""
    train_idx, val_idx, test_idx = stratified_split_indices(
        train_ds.targets, class_names, val_frac, test_frac
    )
    train_subset = Subset(train_ds, train_idx)
    val_subset = Subset(eval_ds, val_idx)
    test_subset = Subset(eval_ds, test_idx)

    if verbose:
        print(f"{reason} (stratified by class, seed 42): {len(train_idx)} train / "
              f"{len(val_idx)} validation / {len(test_idx)} test images.")
        print_class_distribution(train_subset, "train")
        print_class_distribution(val_subset, "validation")
        print_class_distribution(test_subset, "test")

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers)
    return train_loader, val_loader, test_loader, class_names


def _auto_split_dataloaders(root, batch_size, num_workers, augment_train, verbose,
                            val_frac, test_frac):
    """Loaders for a folder-per-class dataset with no existing split."""
    train_ds = datasets.ImageFolder(root, transform=get_transforms(train=augment_train))
    eval_ds = datasets.ImageFolder(root, transform=get_transforms(train=False))
    return _split_into_loaders(
        train_ds, eval_ds, train_ds.classes, batch_size, num_workers, verbose,
        val_frac, test_frac,
        reason=f"No train/test folders found in '{root}'; splitting automatically",
    )


def _resplit_dataloaders(split_dirs, batch_size, num_workers, augment_train, verbose,
                         val_frac, test_frac):
    """Pool every image from the existing split folders and re-split 80/10/10."""
    folders = [datasets.ImageFolder(d) for d in split_dirs]
    pooled, class_names = merge_split_samples(
        [f.samples for f in folders], [f.classes for f in folders]
    )
    train_ds = PooledImageDataset(pooled, class_names, get_transforms(train=augment_train))
    eval_ds = PooledImageDataset(pooled, class_names, get_transforms(train=False))
    return _split_into_loaders(
        train_ds, eval_ds, class_names, batch_size, num_workers, verbose,
        val_frac, test_frac,
        reason=f"Pooled all images from {len(split_dirs)} split folders and re-split",
    )


def get_dataloaders(data_dir=DATA_DIR, batch_size=32, val_split=0.15, num_workers=2,
                    augment_train=True, verbose=True,
                    auto_val_split=0.10, auto_test_split=0.10, resplit=False):
    """Returns train_loader, val_loader, test_loader, class_names.

    What happens depends on what data_dir contains:
      * Training/ + Validation/ + Testing/ : used as provided.
      * Training/ + Testing/ only          : val_split (15%) of Training/ becomes
        the validation set; Testing/ is used only for final evaluation
        (evaluate.py) so numbers stay honest.
      * one folder per class, no split     : split automatically using
        auto_val_split / auto_test_split (fractions of the whole dataset,
        default 80/10/10).
      * resplit=True                       : pool all images from the split folders
        and re-split 80/10/10, ignoring the dataset's own split. Do NOT use this
        on a dataset that was split by patient: it would mix a patient's images
        across splits.

    verbose=False suppresses the class-distribution printout (evaluate.py uses
    this because it writes those counts to CSV instead).
    """
    try:
        train_dir, test_dir = resolve_split_dirs(data_dir)
    except FileNotFoundError as error:
        class_root = find_class_root(data_dir)
        if class_root is None:
            raise error
        return _auto_split_dataloaders(
            class_root, batch_size, num_workers, augment_train, verbose,
            auto_val_split, auto_test_split,
        )

    val_dir = _find_dir(os.path.dirname(train_dir), VAL_NAMES)

    if resplit:
        split_dirs = [d for d in (train_dir, val_dir, test_dir) if d]
        return _resplit_dataloaders(
            split_dirs, batch_size, num_workers, augment_train, verbose,
            auto_val_split, auto_test_split,
        )

    if verbose:
        print(f"Training data:   {train_dir}")
        print(f"Validation data: {val_dir or f'{val_split:.0%} of the training images'}")
        print(f"Testing data:    {test_dir}")

    full_train = datasets.ImageFolder(
        train_dir, transform=get_transforms(train=augment_train)
    )
    class_names = full_train.classes

    if val_dir:
        # The dataset ships its own validation folder: use it as provided.
        train_part = full_train
        val_part = datasets.ImageFolder(val_dir, transform=get_transforms(train=False))
        if val_part.classes != class_names:
            raise ValueError(
                f"Class folders in '{val_dir}' ({val_part.classes}) don't match "
                f"the training folders ({class_names})."
            )
    else:
        n_val = int(len(full_train) * val_split)
        n_train = len(full_train) - n_val
        train_part, val_part = random_split(
            full_train, [n_train, n_val],
            generator=torch.Generator().manual_seed(42)
        )
        # val split should NOT use training augmentation -- swap in eval transforms
        val_part.dataset = datasets.ImageFolder(
            train_dir, transform=get_transforms(train=False)
        )

    test_dataset = datasets.ImageFolder(test_dir, transform=get_transforms(train=False))

    if verbose:
        print_class_distribution(train_part, "train")
        print_class_distribution(val_part, "validation")

    train_loader = DataLoader(train_part, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers)
    val_loader = DataLoader(val_part, batch_size=batch_size, shuffle=False,
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