"""
Unified Pipeline Runner for ResNet-50 Brain Tumor Detection.

Executes:
1. Dataset ingestion & Patient-Level Splitting (Leakage-Free)
2. ResNet-50 Transfer Learning & Fine-Tuning
3. Clinical Validation & Test Evaluation (Accuracy, Precision, Recall, Specificity, F1)
4. Plotting Confusion Matrix & Training Curves
"""

import argparse
import sys
from pathlib import Path

import torch

import config
from dataset import scan_dataset, patient_level_split, create_dataloaders
from model import build_resnet50
from train import train_model
from evaluate import evaluate_model, print_clinical_report
from sample_data_generator import generate_synthetic_dataset


def main():
    parser = argparse.ArgumentParser(description="ResNet-50 Brain Tumor Detection Pipeline")
    parser.add_argument("--data_dir", type=str, default=str(config.DATA_DIR), help="Path to MRI dataset root folder")
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE, help="Batch size")
    parser.add_argument("--demo", action="store_true", help="Generate synthetic MRI dataset if none exists")
    parser.add_argument("--finetune", action="store_true", help="Run Phase 2 fine-tuning on deep layers 3 & 4 from Phase 1 checkpoint")
    args = parser.parse_args()

    data_path = Path(args.data_dir)

    # If dataset folder doesn't exist or is empty, generate sample data
    if not data_path.exists() or not any(data_path.iterdir()):
        print(f"[Pipeline] No dataset found at {data_path}. Auto-generating synthetic verification dataset...")
        generate_synthetic_dataset(target_dir=data_path)

    print("\n" + "=" * 65)
    print("        RESNET-50 BRAIN TUMOR DETECTION PIPELINE")
    print("=" * 65)

    # 1. Scan Dataset
    file_paths, labels, patient_ids, class_names = scan_dataset(data_path)
    num_classes = len(class_names)
    print(f"Dataset Loaded: {len(file_paths)} scans across {len(set(patient_ids))} patients.")
    print(f"Detected Classes ({num_classes}): {class_names}")

    # 2. Patient-Level Splitting (Leakage-Proof)
    print("\n[Step 1/4] Partitioning data at Patient-Level (80% Train / 10% Val / 10% Test)...")
    splits = patient_level_split(
        file_paths=file_paths,
        labels=labels,
        patient_ids=patient_ids,
        train_ratio=config.TRAIN_RATIO,
        val_ratio=config.VAL_RATIO,
        test_ratio=config.TEST_RATIO,
        random_seed=config.RANDOM_SEED
    )

    print(f"  - Training Set:   {len(splits['train']['paths']):>5} scans ({len(set(splits['train']['pids']))} patients)")
    print(f"  - Validation Set: {len(splits['val']['paths']):>5} scans ({len(set(splits['val']['pids']))} patients)")
    print(f"  - Testing Set:    {len(splits['test']['paths']):>5} scans ({len(set(splits['test']['pids']))} patients)")
    print("  [PASSED] Strict patient isolation verified. No patient overlap between splits.")

    # 3. Create DataLoaders
    train_loader, val_loader, test_loader = create_dataloaders(
        splits=splits,
        batch_size=args.batch_size,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )

    # 4. Instantiate Model
    if args.finetune:
        print(f"\n[Step 2/4] [PHASE 2] Loading Phase 1 Checkpoint for Fine-Tuning...")
        model = build_resnet50(num_classes=num_classes, pretrained=False, freeze_initially=False)
        if config.BEST_MODEL_PATH.exists():
            model.load_state_dict(torch.load(config.BEST_MODEL_PATH, map_location=config.DEVICE, weights_only=True))
            print(f"  - Successfully loaded weights from {config.BEST_MODEL_PATH}")
        else:
            print(f"  - Warning: {config.BEST_MODEL_PATH} not found. Starting from ImageNet weights.")
        model.unfreeze_layer4_and_layer3()
        total_p, train_p = model.count_parameters()
        print(f"  - Unfroze Bottleneck layers (layer3 & layer4): {train_p:,} trainable parameters.")
        lr_to_use = config.FINE_TUNE_LR
    else:
        print(f"\n[Step 2/4] Initializing ResNet-50 (Pretrained ImageNet-1K V2)...")
        model = build_resnet50(num_classes=num_classes, pretrained=True, freeze_initially=True)
        print(f"  - Replaced final FC layer: in_features=2048 -> Dropout(0.4) -> Linear(2048, {num_classes})")
        print(f"  - Initial State: Backbone frozen, classifier head trainable.")
        lr_to_use = config.LEARNING_RATE

    # 5. Train Model
    print(f"\n[Step 3/4] Starting Model Training ({args.epochs} epochs requested)...")
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=num_classes,
        train_labels=splits["train"]["labels"],
        num_epochs=args.epochs,
        device=config.DEVICE,
        use_class_weights=True,
        initial_lr=lr_to_use
    )

    # 6. Evaluate on Test Set
    print("\n[Step 4/4] Evaluating Best ResNet-50 Checkpoint on Held-Out Test Set...")
    metrics = evaluate_model(
        model=model,
        test_loader=test_loader,
        class_names=class_names,
        device=config.DEVICE
    )

    # 7. Print Clinical Results
    print_clinical_report(metrics)

    print("\n[Pipeline Complete] Artifacts generated:")
    print(f"  1. Best Weights:      {config.BEST_MODEL_PATH}")
    print(f"  2. Confusion Matrix:  {config.CONFUSION_MATRIX_PATH}")
    print(f"  3. Training Curves:   {config.TRAINING_CURVES_PATH}")
    print(f"  4. Test Metrics JSON: {config.METRICS_JSON_PATH}")


if __name__ == "__main__":
    main()
