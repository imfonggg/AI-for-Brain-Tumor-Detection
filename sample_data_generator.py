"""
Mock MRI Dataset Generator for Pipeline Verification.

Creates a structured set of synthetic brain MRI images organized by class
with realistic Patient IDs (e.g., patient_001_slice_01.png) to test and verify
the entire patient-level data partitioning, model training, and clinical evaluation
pipeline before loading the full 12,000 Kaggle dataset.
"""

from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np

import config


def generate_synthetic_dataset(
    target_dir: Path = config.DATA_DIR,
    classes: list = None,
    patients_per_class: int = 8,
    slices_per_patient: int = 4
) -> None:
    """
    Generates dummy MRI scans simulating head contours and internal signals.
    """
    if classes is None:
        classes = ["glioma", "meningioma", "pituitary", "no_tumor"]

    print(f"[Sample Data] Generating synthetic MRI dataset in: {target_dir}")
    patient_counter = 1

    for class_name in classes:
        class_folder = target_dir / class_name
        class_folder.mkdir(parents=True, exist_ok=True)

        for _ in range(patients_per_class):
            patient_id = f"pat{patient_counter:03d}"
            patient_counter += 1

            for slice_idx in range(1, slices_per_patient + 1):
                filename = f"{patient_id}_slice{slice_idx:02d}.png"
                filepath = class_folder / filename

                # Create synthetic 224x224 brain slice image
                img = Image.new("L", (224, 224), color=10)
                draw = ImageDraw.Draw(img)

                # Outer skull ellipse
                draw.ellipse([25, 20, 199, 204], outline=140, fill=30)
                # Brain tissue ellipse
                draw.ellipse([35, 30, 189, 194], outline=80, fill=60)
                # Ventricles / internal structure
                draw.ellipse([95, 80, 129, 144], outline=40, fill=20)

                # Simulated tumor signal if not 'no_tumor'
                if class_name != "no_tumor":
                    # Randomize tumor location and brightness per patient
                    draw.ellipse([130, 60, 170, 100], fill=210)

                img.save(filepath)

    total_images = len(classes) * patients_per_class * slices_per_patient
    total_patients = len(classes) * patients_per_class
    print(f"[Sample Data] Created {total_images} images across {total_patients} patients in {len(classes)} classes.")


if __name__ == "__main__":
    generate_synthetic_dataset()
