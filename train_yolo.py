"""
Optimized YOLOv8 Training for Mac (Apple Silicon MPS)
======================================================
Speed optimizations applied:
  - imgsz=416   → matches actual image size, no upscaling overhead (was 640)
  - batch=32    → higher batch = fewer gradient steps per epoch (was 16)
  - workers=4   → parallel data loading (was default 8, which thrashes MPS)
  - cache=True  → caches images in RAM after first epoch, hugely speeds up subsequent epochs
  - patience=10 → early-stop if no improvement for 10 epochs (skip wasted time)
  - amp=False   → MPS doesn't support FP16 well; keep FP32 to avoid NaN losses
  - optimizer='AdamW' + lr0=0.001 → converges faster than default SGD on small datasets

Expected time: ~1–2 hours for 30 epochs on M1/M2/M3 Mac (down from 20-30 hrs).
"""

from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")   # Nano — smallest & fastest, best for Mac

    print("Starting Optimized YOLOv8 training for Stain Detection...")
    print("Expected time: ~1-2 hours on Apple Silicon\n")

    results = model.train(
        data="stain_data.yaml",

        # ── Core ──────────────────────────────────────────────────
        epochs=30,
        imgsz=416,          # ✅ Matches your actual image size (was 640 = big slowdown)
        batch=32,           # ✅ Larger batch = faster epoch (was 16)
        device="mps",       # Apple Silicon GPU

        # ── Speed ─────────────────────────────────────────────────
        workers=4,          # ✅ Parallel data loading without thrashing MPS
        cache=True,         # ✅ Cache images in RAM — huge speedup after epoch 1
        amp=False,          # ✅ MPS FP16 is buggy; keep FP32 for stable training

        # ── Optimizer ─────────────────────────────────────────────
        optimizer="AdamW",  # ✅ Converges faster than SGD on small datasets
        lr0=0.001,          # Good starting LR for AdamW
        weight_decay=0.0005,

        # ── Early Stopping ────────────────────────────────────────
        patience=10,        # ✅ Stop early if mAP stops improving (saves hours)

        # ── Output ────────────────────────────────────────────────
        project="runs/detect/stain_detection",
        name="yolov8_stains",
        exist_ok=True,
        verbose=True,
    )

    print("\n✅ Training complete!")
    print("   Best weights: runs/detect/stain_detection/yolov8_stains/weights/best.pt")
    print("   Run: python3 predict_yolo.py <image_path>")

if __name__ == "__main__":
    main()
