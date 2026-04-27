"""
Predict stain type from a fabric image.
Usage: python predict.py path/to/image.jpg
"""

import argparse
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pathlib import Path


BASE_DIR = Path("/Users/rishabhgosain/Downloads/FabricSpotDefect An Annotated Dataset for Identifying Spot  Defects in Different Fabric Types/FabricSpotDefect")
MODEL_FILE = BASE_DIR / "stain_classifier_best.pkl"
MAPPING_FILE = BASE_DIR / "class_mapping.json"


def get_feature_extractor(device):
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    model.classifier = nn.Identity()
    model = model.to(device)
    model.eval()
    return model


def extract_single_image(img_path, model, device):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img = Image.open(img_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model(tensor).cpu().numpy()
    return features


def main():
    parser = argparse.ArgumentParser(description="Classify fabric stain type.")
    parser.add_argument("image_path", type=str, help="Path to the image file")
    args = parser.parse_args()

    if not MODEL_FILE.exists():
        print(f"Model not found at {MODEL_FILE}.\nPlease run train_classifier.py first.")
        return

    with open(MAPPING_FILE) as f:
        class_names = json.load(f)

    with open(MODEL_FILE, "rb") as f:
        clf = pickle.load(f)

    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")

    print("Loading feature extractor...")
    feat_model = get_feature_extractor(device)

    print(f"Processing image: {args.image_path}")
    features = extract_single_image(args.image_path, feat_model, device)

    probs = clf.predict_proba(features)[0]
    predicted_idx = np.argmax(probs)
    predicted_class = class_names[predicted_idx]

    print(f"\n{'='*40}")
    print(f"  Prediction: {predicted_class.upper().replace('_', ' ')}")
    print(f"{'='*40}")
    print("\nConfidence Scores:")
    for i, name in enumerate(class_names):
        bar = "█" * int(probs[i] * 30)
        print(f"  {name:15s}: {probs[i]*100:5.1f}%  {bar}")


if __name__ == "__main__":
    main()
