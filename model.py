"""
Model definitions.

- BaselineCNN: small CNN trained from scratch. Use this as your "before" number.
- get_pretrained_model(): transfer-learning models (resnet18, resnet50,
  efficientnet_b0) with the final layer swapped for your number of classes.
  This is what the "explore different ML/DL approaches" guideline wants you
  to compare against the baseline.
"""

import torch.nn as nn
from torchvision import models


class BaselineCNN(nn.Module):
    """A simple CNN from scratch, deliberately unfancy so it's a fair baseline
    to compare transfer learning against."""

    def __init__(self, num_classes=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 224 -> 112

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 112 -> 56

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 56 -> 28

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 28 -> 14
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def get_pretrained_model(name: str, num_classes=4, freeze_backbone=False):
    """Loads a torchvision model pretrained on ImageNet and swaps the head.

    freeze_backbone=True only trains the new final layer (fast, less prone to
    overfitting on small datasets). freeze_backbone=False fine-tunes the whole
    network (usually better accuracy if you have enough data + epochs).
    """
    name = name.lower()

    if name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    else:
        raise ValueError(
            f"Unknown model '{name}'. Choose from: baseline, resnet18, resnet50, efficientnet_b0"
        )

    return model


def build_model(name: str, num_classes=4, freeze_backbone=False):
    """Single entry point used by train.py / evaluate.py."""
    if name.lower() == "baseline":
        return BaselineCNN(num_classes=num_classes)
    return get_pretrained_model(name, num_classes=num_classes, freeze_backbone=freeze_backbone)
