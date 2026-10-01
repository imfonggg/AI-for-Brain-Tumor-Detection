"""Compare a trained ResNet50 softmax head with an RBF-SVM on its features.

Usage:
    python hybrid_svm.py --checkpoint outputs/resnet50_best.pt
"""

import argparse
import os

import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm

from data_loader import get_dataloaders
from model import build_resnet50

OUTPUT_DIR = "outputs"


def predict_softmax(model, loader, device):
    model.eval()
    predictions, labels = [], []
    with torch.no_grad():
        for images, batch_labels in tqdm(loader, desc="ResNet50 softmax"):
            logits = model(images.to(device))
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            labels.extend(batch_labels.tolist())
    return predictions, labels


def extract_features(model, loader, device):
    model.eval()
    features, labels = [], []
    with torch.no_grad():
        for images, batch_labels in tqdm(loader, desc="Extracting features"):
            batch_features = model(images.to(device))
            features.append(batch_features.cpu())
            labels.append(batch_labels)
    return torch.cat(features).numpy(), torch.cat(labels).numpy()


def report_results(name, labels, predictions, class_names):
    print(f"\n=== {name} on Testing/ set ===")
    print(f"Accuracy: {accuracy_score(labels, predictions):.4f}")
    print(classification_report(
        labels, predictions, labels=range(len(class_names)),
        target_names=class_names, zero_division=0,
    ))
    matrix = confusion_matrix(labels, predictions, labels=range(len(class_names)))
    plt.figure(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix: {name}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{name}_confusion_matrix.png"), dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, _, test_loader, class_names = get_dataloaders(
        data_dir=args.data_dir, batch_size=args.batch_size, augment_train=False,
    )
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if checkpoint.get("model_name") != "resnet50":
        raise ValueError("The checkpoint must be from a trained ResNet50 model.")
    class_names = checkpoint.get("class_names", class_names)

    model = build_resnet50(num_classes=len(class_names), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    softmax_predictions, test_labels = predict_softmax(model, test_loader, device)
    report_results("resnet50_softmax", test_labels, softmax_predictions, class_names)

    model.fc = nn.Identity()
    train_features, train_labels = extract_features(model, train_loader, device)
    test_features, _ = extract_features(model, test_loader, device)

    classifier = make_pipeline(StandardScaler(), SVC(kernel="rbf"))
    classifier.fit(train_features, train_labels)
    svm_predictions = classifier.predict(test_features)
    report_results("resnet50_svm", test_labels, svm_predictions, class_names)

    model_path = os.path.join(OUTPUT_DIR, "resnet50_svm.joblib")
    joblib.dump(classifier, model_path)
    print(f"SVM pipeline saved to {model_path}")


if __name__ == "__main__":
    main()
