"""Create a Grad-CAM overlay for one image using a trained ResNet50 checkpoint.

Usage:
    python grad_cam.py --checkpoint outputs/resnet50_best.pt --image path/to/mri.jpg
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from data_loader import get_transforms
from model import build_resnet50


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", default="outputs/gradcam.png")
    parser.add_argument("--target_class", default=None,
                        help="Class name or class index; defaults to the predicted class")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if checkpoint.get("model_name") != "resnet50":
        raise ValueError("The checkpoint must be from a trained ResNet50 model.")
    class_names = checkpoint["class_names"]

    model = build_resnet50(num_classes=len(class_names), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    original = Image.open(args.image).convert("RGB").resize((224, 224))
    image_tensor = get_transforms(train=False)(original).unsqueeze(0).to(device)
    activations = {}
    gradients = {}

    def capture_activation(_module, _inputs, output):
        activations["value"] = output
        output.register_hook(lambda gradient: gradients.setdefault("value", gradient))

    hook = model.layer4[-1].register_forward_hook(capture_activation)
    model.zero_grad(set_to_none=True)
    logits = model(image_tensor)
    predicted_index = int(logits.argmax(dim=1).item())

    if args.target_class is None:
        target_index = predicted_index
    elif args.target_class in class_names:
        target_index = class_names.index(args.target_class)
    else:
        target_index = int(args.target_class)
        if target_index < 0 or target_index >= len(class_names):
            raise ValueError(f"Target index must be between 0 and {len(class_names) - 1}.")

    logits[0, target_index].backward()
    hook.remove()

    feature_map = activations["value"]
    gradient = gradients["value"]
    weights = gradient.mean(dim=(2, 3), keepdim=True)
    cam = (weights * feature_map).sum(dim=1, keepdim=True).relu()
    cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)[0, 0]
    cam = cam.detach().cpu().numpy()
    cam -= cam.min()
    if cam.max() > 0:
        cam /= cam.max()

    heatmap = (plt.get_cmap("jet")(cam)[..., :3] * 255).astype(np.uint8)
    overlay = Image.blend(original, Image.fromarray(heatmap), alpha=0.4)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    overlay.save(args.output)

    confidence = torch.softmax(logits, dim=1)[0, target_index].item()
    print(f"Predicted: {class_names[predicted_index]}")
    print(f"Grad-CAM target: {class_names[target_index]} (confidence={confidence:.4f})")
    print(f"Heatmap saved to {args.output}")


if __name__ == "__main__":
    main()
