"""
Training Engine for ResNet-50 Brain Tumor Detection.

Features:
- Two-Stage Transfer Learning (Backbone Freezing -> Deep Layer Fine-Tuning)
- Cross-Entropy Loss with optional Class Weighting for class balance (Shraddaya's topic)
- AdamW Optimizer with Learning Rate Scheduling
- Validation monitoring, best checkpoint saving, and early stopping
- Training curves plotting (Loss & Accuracy)
"""

import copy
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from model import BrainTumorResNet50


def calculate_class_weights(train_labels: List[int], num_classes: int) -> torch.Tensor:
    """
    Computes inverse frequency class weights to penalize underrepresented tumor types.
    Assists Shraddaya's loss function research on handling imbalanced medical data.
    """
    counts = np.bincount(train_labels, minlength=num_classes)
    total = len(train_labels)
    # Avoid zero division
    weights = total / (num_classes * np.maximum(counts, 1).astype(float))
    return torch.tensor(weights, dtype=torch.float32)


def plot_training_curves(history: Dict[str, List[float]], save_path: Path) -> None:
    """
    Plots training vs validation loss and accuracy across epochs.
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss plot
    ax1.plot(epochs, history["train_loss"], "o-", label="Train Loss", color="#1f77b4")
    ax1.plot(epochs, history["val_loss"], "s-", label="Val Loss", color="#ff7f0e")
    ax1.set_title("Cross-Entropy Loss vs. Epochs", fontsize=13)
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Loss", fontsize=11)
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Accuracy plot
    ax2.plot(epochs, [a * 100 for a in history["train_acc"]], "o-", label="Train Accuracy", color="#2ca02c")
    ax2.plot(epochs, [a * 100 for a in history["val_acc"]], "s-", label="Val Accuracy", color="#d62728")
    ax2.set_title("Classification Accuracy vs. Epochs", fontsize=13)
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Accuracy (%)", fontsize=11)
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Training] Training curves saved to: {save_path}")


def train_model(
    model: BrainTumorResNet50,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_classes: int,
    train_labels: List[int],
    num_epochs: int = config.NUM_EPOCHS,
    device: torch.device = config.DEVICE,
    use_class_weights: bool = True,
    initial_lr: float = config.LEARNING_RATE
) -> Tuple[BrainTumorResNet50, Dict[str, List[float]]]:
    """
    Executes the training and validation loop.
    """
    model.to(device)

    # Loss function with optional class weighting
    if use_class_weights:
        weights = calculate_class_weights(train_labels, num_classes).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        print(f"[Loss Function] Using Class-Weighted Cross-Entropy Loss.")
    else:
        criterion = nn.CrossEntropyLoss()
        print(f"[Loss Function] Using Standard Cross-Entropy Loss.")

    # Optimizer: Start with training active parameters
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=initial_lr,
        weight_decay=config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    best_val_loss = float("inf")
    best_model_weights = copy.deepcopy(model.state_dict())
    patience_counter = 0

    print("\n" + "=" * 65)
    print(f"Starting Training on Device: {device}")
    total_p, train_p = model.count_parameters()
    print(f"Model Parameters: {total_p:,} total | {train_p:,} trainable initially")
    print("=" * 65)

    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        # Phase 2 Fine-Tuning Transition Check
        if config.FINE_TUNE and epoch == config.FINE_TUNE_START_EPOCH:
            print(f"\n>>> [Transfer Learning] Unfreezing ResNet-50 layer3 and layer4 for fine-tuning...")
            model.unfreeze_layer4_and_layer3()
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=config.FINE_TUNE_LR,
                weight_decay=config.WEIGHT_DECAY
            )
            _, train_p = model.count_parameters()
            print(f">>> Trainable parameters updated to: {train_p:,}")

        # --- TRAINING PHASE ---
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_train_samples = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{num_epochs:02d} [Train]")
        for images, labels, _ in pbar:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * images.size(0)
            running_corrects += torch.sum(preds == labels.data).item()
            total_train_samples += images.size(0)

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        epoch_train_loss = running_loss / total_train_samples
        epoch_train_acc = running_corrects / total_train_samples

        # --- VALIDATION PHASE ---
        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0
        total_val_samples = 0

        with torch.no_grad():
            for images, labels, _ in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)
                val_running_loss += loss.item() * images.size(0)
                val_running_corrects += torch.sum(preds == labels.data).item()
                total_val_samples += images.size(0)

        epoch_val_loss = val_running_loss / total_val_samples
        epoch_val_acc = val_running_corrects / total_val_samples

        scheduler.step(epoch_val_loss)

        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)

        print(
            f"Epoch {epoch:02d} Summary: "
            f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc * 100:.2f}% | "
            f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc * 100:.2f}%"
        )

        # Checkpoint if validation loss improves
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), config.BEST_MODEL_PATH)
            print(f"  --> Checkpoint saved: New best Val Loss {best_val_loss:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                print(f"\n[Early Stopping] No validation loss improvement for {config.EARLY_STOPPING_PATIENCE} epochs.")
                break

    time_elapsed = time.time() - start_time
    print(f"\nTraining completed in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best Validation Loss: {best_val_loss:.4f}")

    # Plot training and validation loss/accuracy curves
    plot_training_curves(history, config.TRAINING_CURVES_PATH)

    # Load best weights before returning
    model.load_state_dict(best_model_weights)
    return model, history
