# 🧵 Fabric Spot Defect Classifier — How to Run

A two-pipeline system for detecting and classifying fabric stains (oil, dirt, grease):
- **Pipeline 1:** EfficientNet-B0 + SVM — fast classification (~80% accuracy)
- **Pipeline 2:** YOLOv8 — object detection with bounding boxes

---

## 📋 Requirements

### Python Version
Python 3.9+

### Install Dependencies
```bash
pip install torch torchvision tqdm scikit-learn pillow ultralytics
```

---

## 📁 Dataset Structure

Your `SeparatedDataset/` folder must follow this layout (used by Pipeline 1):
```
SeparatedDataset/
├── dirt_stain/
│   ├── img1.jpg
│   └── ...
├── grease_stain/
│   └── ...
└── oil_stain/
    └── ...
```

Your `Augmented/YOLOv8/` folder must follow this layout (used by Pipeline 2):
```
Augmented/YOLOv8/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

---

## 🚀 Pipeline 1 — EfficientNet-B0 + SVM Classifier

### Option A: Run Locally (Mac)

**Step 1: Train the classifier**
```bash
python train_classifier.py
```
- Extracts 1280-dim features using frozen EfficientNet-B0
- Trains an RBF-SVM on top
- Saves `stain_classifier_best.pkl` and `class_mapping.json`
- Features are cached in `features.npz` (re-running skips extraction)

**Step 2: Predict on a new image**
```bash
python predict.py path/to/your/image.jpg
```

---

### Option B: Run on Google Colab (Recommended — GPU is faster)

1. Upload `SeparatedDataset/` to Google Drive at:
   ```
   MyDrive/FabricSpotDefect/SeparatedDataset/
   ```
2. Open `colab.ipynb` at [colab.research.google.com](https://colab.research.google.com)
3. Set **Runtime → Change runtime type → T4 GPU**
4. Update `BASE_DIR` in Cell 3 to match your Drive path
5. Run all cells top to bottom ▶️

> **Tip:** Upload `features.npz` to Drive alongside `SeparatedDataset/` to skip the ~5 min extraction step on future runs.

---

## 🎯 Pipeline 2 — YOLOv8 Object Detection

### Step 1: Update `stain_data.yaml`
Edit the `path` field to point to your local `Augmented/YOLOv8/` directory:
```yaml
path: /your/local/path/to/FabricSpotDefect/Augmented/YOLOv8
train: train/images
val: valid/images
test: test/images

names:
  0: dirt_stain
  1: grease_stain
  2: oil_stain
```

### Step 2: Train YOLOv8
```bash
python train_yolo.py
```
- Uses YOLOv8 Nano (`yolov8n.pt`) — fastest model
- Optimized for Apple Silicon (MPS) by default
- Best weights saved to: `runs/detect/stain_detection/yolov8_stains/weights/best.pt`
- Expected time: ~1–2 hours on Apple Silicon

### Step 3: Run Detection on a New Image
```bash
python predict_yolo.py path/to/your/image.jpg
```

---

## 📊 Results (Pipeline 1 — SVM)

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| dirt_stain | 0.78 | 0.85 | 0.82 |
| grease_stain | 0.81 | 0.66 | 0.73 |
| oil_stain | 0.83 | 0.86 | 0.84 |
| **Overall** | **0.81** | **0.79** | **0.80** |

- **Validation Accuracy:** 79.60%
- **Test Accuracy:** 80.31%

---

## 📂 Key Files

| File | Description |
|------|-------------|
| `train_classifier.py` | Train EfficientNet-B0 + SVM (local) |
| `colab.ipynb` | Train EfficientNet-B0 + SVM (Google Colab) |
| `predict.py` | Run inference with SVM model |
| `train_yolo.py` | Train YOLOv8 detection model |
| `predict_yolo.py` | Run inference with YOLOv8 model |
| `sort_augmented.py` | Sort/augment dataset images |
| `stain_data.yaml` | YOLOv8 dataset config |
| `stain_classifier_best.pkl` | Trained SVM model *(not in repo — generate locally)* |
| `features.npz` | Cached EfficientNet features *(not in repo — generate locally)* |

---

## ⚠️ Notes

- `stain_classifier_best.pkl` and `features.npz` are excluded from Git (too large). Generate them by running `train_classifier.py` or `colab.ipynb`.
- Update absolute paths in `predict.py` and `stain_data.yaml` to match your local directory before running locally.
- For Colab, always update `BASE_DIR` in Cell 3 of `colab.ipynb` to match your Google Drive folder path.
