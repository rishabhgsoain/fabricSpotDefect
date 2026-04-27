"""
Fast Stain Classifier — Feature Extraction + Sklearn
======================================================
Instead of full fine-tuning (slow), we:
1. Use EfficientNet-B0 as a frozen feature extractor (~5 min to extract all features)
2. Train a LogisticRegression classifier on those features (< 5 seconds)
Total time: ~5 minutes vs hours of fine-tuning.
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import pickle


BASE_DIR = Path("/Users/rishabhgosain/Downloads/FabricSpotDefect An Annotated Dataset for Identifying Spot  Defects in Different Fabric Types/FabricSpotDefect")
DATA_DIR = BASE_DIR / "SeparatedDataset"
FEATURES_FILE = BASE_DIR / "features.npz"
MODEL_FILE = BASE_DIR / "stain_classifier_best.pkl"
MAPPING_FILE = BASE_DIR / "class_mapping.json"


def get_feature_extractor(device):
    """Load EfficientNet-B0 with the classification head removed (feature extractor only)."""
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    # Remove the classifier head — output is a 1280-dim feature vector
    model.classifier = nn.Identity()
    model = model.to(device)
    model.eval()
    return model


def extract_features(data_dir, device):
    """Extract 1280-dim feature vectors for all images using EfficientNet-B0."""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder(data_dir, transform=transform)
    class_names = dataset.classes
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)

    print(f"Classes: {class_names}")
    print(f"Total images: {len(dataset)}")

    model = get_feature_extractor(device)

    all_features = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Extracting features"):
            images = images.to(device)
            feats = model(images)  # shape: (batch, 1280)
            all_features.append(feats.cpu().numpy())
            all_labels.append(labels.numpy())

    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    return all_features, all_labels, class_names


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # --- Step 1: Extract or load features ---
    if FEATURES_FILE.exists():
        print(f"Loading cached features from {FEATURES_FILE}...")
        data = np.load(FEATURES_FILE, allow_pickle=True)
        X = data["features"]
        y = data["labels"]
        class_names = list(data["class_names"])
        print(f"Loaded {len(X)} feature vectors.")
    else:
        print("Extracting features with EfficientNet-B0 (frozen)...")
        X, y, class_names = extract_features(DATA_DIR, device)
        np.savez(FEATURES_FILE, features=X, labels=y, class_names=class_names)
        print(f"Saved features to {FEATURES_FILE}")

    # Save class mapping for predict.py
    with open(MAPPING_FILE, "w") as f:
        json.dump(class_names, f)

    # --- Step 2: Split into train/val/test ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.125, random_state=42, stratify=y_train
    )
    print(f"\nSplit: Train={len(X_train)}  Val={len(X_val)}  Test={len(X_test)}")

    # --- Step 3: Train SVM Classifier ---
    print("\nTraining Support Vector Machine (SVM) classifier...")
    clf = SVC(kernel='rbf', C=10.0, gamma='scale', probability=True, random_state=42, verbose=True)
    clf.fit(X_train, y_train)

    # --- Step 4: Evaluate ---
    val_preds = clf.predict(X_val)
    val_acc = accuracy_score(y_val, val_preds)
    print(f"\nValidation Accuracy: {val_acc*100:.2f}%")

    test_preds = clf.predict(X_test)
    test_acc = accuracy_score(y_test, test_preds)
    print(f"Test Accuracy:       {test_acc*100:.2f}%")

    print("\n--- Classification Report (Test Set) ---")
    print(classification_report(y_test, test_preds, target_names=class_names))

    print("--- Confusion Matrix ---")
    cm = confusion_matrix(y_test, test_preds)
    print(f"       {class_names}")
    for i, row in enumerate(cm):
        print(f"  {class_names[i][:10]:12s} | {row}")

    # --- Step 5: Save the classifier ---
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(clf, f)
    print(f"\nClassifier saved to: {MODEL_FILE}")
    print("Done! Run: python predict.py <image_path>")


if __name__ == "__main__":
    main()
