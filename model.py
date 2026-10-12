"""
ResNet-50 Deep Learning Model Architecture.

Architecture Details for Team Master Doc:
- ResNet-50 utilizes 50 layers composed of 3-layer Bottleneck Blocks (1x1 -> 3x3 -> 1x1 convs).
- In contrast to ResNet-18 (which uses 2-layer Basic Blocks), ResNet-50 has ~25.6M parameters
  giving it greater capacity to detect subtle, minute MRI boundaries and heterogeneous enhancement.
- Residual skip connections pass the identity mapping F(x) + x, preventing vanishing gradients.
- Uses Transfer Learning (ImageNet-pretrained) with fine-tuning capability.
"""

from typing import List, Tuple
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class BrainTumorResNet50(nn.Module):
    """
    ResNet-50 classifier customized for Brain Tumor MRI classification.
    Replaces ImageNet 1000-class head with a custom dropout-regularized classifier.
    """
    def __init__(self, num_classes: int, pretrained: bool = True, dropout_rate: float = 0.4):
        super().__init__()
        self.num_classes = num_classes

        # Load weights
        if pretrained:
            weights = ResNet50_Weights.IMAGENET1K_V2
            self.resnet = resnet50(weights=weights)
        else:
            self.resnet = resnet50(weights=None)

        # In ResNet-50, the output of the final bottleneck block has 2048 features
        # (compared to 512 in ResNet-18).
        in_features = self.resnet.fc.in_features  # 2048

        # Replace classification head with regularized linear projection
        self.resnet.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resnet(x)

    def freeze_backbone(self) -> None:
        """
        Freezes all convolutional backbone layers (conv1, bn1, layer1, layer2, layer3, layer4).
        Only the newly attached classifier head (self.resnet.fc) will receive gradients.
        """
        for name, param in self.resnet.named_parameters():
            if "fc" not in name:
                param.requires_grad = False

    def unfreeze_layer4_and_layer3(self) -> None:
        """
        Unfreezes the deepest residual bottleneck blocks (layer3 and layer4).
        These high-level layers specialize in tumor boundary, shape, and contrast textures.
        """
        for name, param in self.resnet.named_parameters():
            if "layer4" in name or "layer3" in name or "fc" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def unfreeze_all(self) -> None:
        """Unfreezes all parameters across the entire network for full fine-tuning."""
        for param in self.parameters():
            param.requires_grad = True

    def count_parameters(self) -> Tuple[int, int]:
        """Returns (total_params, trainable_params)."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


def build_resnet50(num_classes: int, pretrained: bool = True, freeze_initially: bool = True) -> BrainTumorResNet50:
    """
    Factory function to instantiate and prepare the ResNet-50 model.
    """
    model = BrainTumorResNet50(num_classes=num_classes, pretrained=pretrained)
    if freeze_initially:
        model.freeze_backbone()
    return model
