"""Pretrained model construction for four-class MRI classification."""

import torch.nn as nn
from torchvision import models


def build_resnet18(num_classes=4, pretrained=True):
    """Build ResNet18 and replace its ImageNet classifier with a task head."""
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_resnet50(num_classes=4, pretrained=True):
    """Build ResNet50 and replace its ImageNet classifier with a task head."""
    weights = models.ResNet50_Weights.DEFAULT if pretrained else None
    model = models.resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_efficientnet_b0(num_classes=4, pretrained=True):
    """Build EfficientNet-B0 and replace its ImageNet classifier with a task head.

    EfficientNet-B0's classifier is `nn.Sequential(Dropout, Linear(1280, 1000))`,
    so we only swap out the Linear layer (index 1) and keep the dropout.
    """
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


# Registry of available architectures, keyed by the --model flag used in
# train.py / evaluate.py. 
MODEL_BUILDERS = {
    "resnet18": build_resnet18,
    "resnet50": build_resnet50,
    "efficientnet_b0": build_efficientnet_b0,
}

# Name of each architecture's classifier-head submodule. train.py uses this
# to know which parameters stay trainable during the head-only phase. Both
# ResNets call it `fc`; EfficientNet-B0 calls it `classifier`.
CLASSIFIER_ATTR = {
    "resnet18": "fc",
    "resnet50": "fc",
    "efficientnet_b0": "classifier",
}


def build_model(model_name, num_classes=4, pretrained=True):
    """Build a model by name. See MODEL_BUILDERS for the available names."""
    if model_name not in MODEL_BUILDERS:
        raise ValueError(
            f"Unknown model '{model_name}'. Choose from: {sorted(MODEL_BUILDERS)}"
        )
    return MODEL_BUILDERS[model_name](num_classes=num_classes, pretrained=pretrained)