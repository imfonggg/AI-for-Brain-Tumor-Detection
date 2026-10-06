"""Train the two-stage MRI classifier (ResNet50 or EfficientNet-B0)."""

import argparse
import os
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm

from data_loader import get_dataloaders
from model import build_model, CLASSIFIER_ATTR

OUTPUT_DIR = "outputs"


def run_epoch(model, loader, criterion, optimizer, device, train: bool,
              head_module=None, freeze_backbone: bool = False):
    if train:
        model.train()
        if freeze_backbone:
            # Backbone stays in eval mode so its batch-norm running stats
            # don't update while frozen; only the classifier head trains.
            model.eval()
            head_module.train()
    else:
        model.eval()

    total_loss, correct, total = 0.0, 0, 0
    context = torch.enable_grad() if train else torch.no_grad()

    with context:
        for images, labels in tqdm(loader, leave=False):
            images, labels = images.to(device), labels.to(device)

            if train:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="resnet50",
                         choices=sorted(CLASSIFIER_ATTR),
                         help="Which architecture to train")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--head_epochs", type=int, default=5,
                         help="Epochs with only the classifier head trainable")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--data_dir", default="data")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, _, class_names = get_dataloaders(
        data_dir=args.data_dir, batch_size=args.batch_size
    )
    print(f"Classes: {class_names}")
    print(f"Model: {args.model}")

    criterion = nn.CrossEntropyLoss()
    model = build_model(args.model, num_classes=len(class_names)).to(device)
    head_module = getattr(model, CLASSIFIER_ATTR[args.model])

    phases = [("head", args.head_epochs, 1e-3, True),
              ("fine-tuning", args.epochs, 1e-5, False)]

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    checkpoint_path = os.path.join(OUTPUT_DIR, f"{args.model}_best.pt")

    total_epochs = sum(phase[1] for phase in phases)
    epoch_number = 0

    for phase_name, phase_epochs, learning_rate, freeze_backbone in phases:
        for parameter in model.parameters():
            parameter.requires_grad = not freeze_backbone
        if freeze_backbone:
            for parameter in head_module.parameters():
                parameter.requires_grad = True

        optimizer = torch.optim.Adam(
            filter(lambda parameter: parameter.requires_grad, model.parameters()),
            lr=learning_rate,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2
        )

        print(f"\nPhase: {phase_name} ({phase_epochs} epochs, lr={learning_rate:g})")

        for phase_epoch in range(1, phase_epochs + 1):
            epoch_number += 1
            start = time.time()

            train_loss, train_acc = run_epoch(
                model, train_loader, criterion, optimizer, device, train=True,
                head_module=head_module, freeze_backbone=freeze_backbone,
            )
            val_loss, val_acc = run_epoch(
                model, val_loader, criterion, optimizer, device, train=False,
            )
            scheduler.step(val_loss)

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            elapsed = time.time() - start
            print(f"Epoch {epoch_number}/{total_epochs} ({elapsed:.0f}s) | "
                  f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "class_names": class_names,
                    "model_name": args.model,
                    "val_acc": val_acc,
                }, checkpoint_path)
                print(f"  -> saved new best checkpoint ({val_acc:.4f} val acc)")

    print(f"\nBest validation accuracy: {best_val_acc:.4f}")
    print(f"Checkpoint saved to {checkpoint_path}")

    # Plot training curves
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history["train_acc"], label="train")
    axes[1].plot(history["val_acc"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.suptitle(f"Training curves: {args.model}")
    fig.tight_layout()
    curves_path = os.path.join(OUTPUT_DIR, f"{args.model}_curves.png")
    fig.savefig(curves_path, dpi=150)
    print(f"Training curves saved to {curves_path}")


if __name__ == "__main__":
    main()