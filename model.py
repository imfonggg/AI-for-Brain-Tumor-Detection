"""ResNet50 model construction for four-class MRI classification."""

import torch.nn as nn
from torchvision import models


def build_resnet50(num_classes=4, pretrained=True):
    """Build ResNet50 and replace its ImageNet classifier with a task head."""
    weights = models.ResNet50_Weights.DEFAULT if pretrained else None
    model = models.resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model