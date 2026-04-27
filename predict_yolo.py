import argparse
from ultralytics import YOLO
from pathlib import Path
from PIL import Image

def main():
    parser = argparse.ArgumentParser(description="Predict stain type and location using YOLOv8.")
    parser.add_argument("image_path", type=str, help="Path to the fabric image")
    args = parser.parse_args()

    BASE_DIR = Path(__file__).parent
    # The weights will be saved here after training completes
    weights_path = BASE_DIR / "runs" / "detect" / "stain_detection" / "yolov8_stains" / "weights" / "best.pt"
    
    if not weights_path.exists():
        print(f"ERROR: Model weights not found at {weights_path}.")
        print("Please wait for train_yolo.py to finish training!")
        return

    print("Loading Trained YOLOv8 Stain Detector...")
    model = YOLO(weights_path)
    
    print(f"Running inference on {args.image_path}...")
    # Run prediction
    results = model.predict(
        source=args.image_path,
        conf=0.25,  # Real confidence threshold for a fully trained model
        save=True,  # Automatically save the image with bounding boxes drawn on it
        project="stain_predictions",
        name="test_results",
        exist_ok=True
    )
    
    # Process results
    for r in results:
        boxes = r.boxes
        if len(boxes) == 0:
            print("\nNo stains detected in confident bounds.")
            continue
            
        print(f"\n--- Detected {len(boxes)} stain(s) ---")
        for box in boxes:
            cls_id = int(box.cls[0].item())
            class_name = model.names[cls_id]
            confidence = box.conf[0].item() * 100
            
            # Print bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            print(f"[{class_name.upper()}] Confidence: {confidence:.1f}% | Box: ({int(x1)}, {int(y1)}) to ({int(x2)}, {int(y2)})")
            
    print(f"\nSaved image with bounding boxes drawn over it to:")
    print(f"  stain_predictions/test_results/{Path(args.image_path).name}")

if __name__ == "__main__":
    main()
