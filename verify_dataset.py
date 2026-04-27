import os
import shutil
import torch
import json
from pathlib import Path
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

def main():
    base_dir = Path("/Users/rishabhgosain/Downloads/FabricSpotDefect An Annotated Dataset for Identifying Spot  Defects in Different Fabric Types/FabricSpotDefect")
    separated_dir = base_dir / "SeparatedDataset"
    categories = ["oil stain", "dirt stain", "grease stain"]
    folder_names = {
        "oil stain": "oil_stain",
        "dirt stain": "dirt_stain",
        "grease stain": "grease_stain"
    }
    folder_to_cat = {v: k for k, v in folder_names.items()}
    confidence_threshold = 0.5  # flag anything below 50% confidence

    print("Loading CLIP model...")
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    text_prompts = [
        "a close up photo of fabric with an oil stain",
        "a close up photo of fabric with a dirt stain",
        "a close up photo of fabric with a grease stain",
    ]
    text_inputs = processor(text=text_prompts, return_tensors="pt", padding=True).to(device)

    misclassified = []
    low_confidence = []
    correct = 0
    total = 0
    stats = {cat: {"correct": 0, "wrong": 0, "low_conf": 0} for cat in categories}

    print("\nVerifying all images...\n")

    for folder_name in ["oil_stain", "dirt_stain", "grease_stain"]:
        folder_path = separated_dir / folder_name
        current_cat = folder_to_cat[folder_name]
        image_files = list(folder_path.glob("*.jpg")) + list(folder_path.glob("*.jpeg")) + list(folder_path.glob("*.png"))

        for img_path in tqdm(image_files, desc=f"Checking {folder_name}"):
            try:
                image = Image.open(img_path).convert("RGB")
                inputs = processor(images=image, return_tensors="pt").to(device)
                inputs['input_ids'] = text_inputs['input_ids']
                inputs['attention_mask'] = text_inputs['attention_mask']

                with torch.no_grad():
                    outputs = model(**inputs)

                probs = outputs.logits_per_image.softmax(dim=1)[0]
                predicted_idx = probs.argmax().item()
                predicted_cat = categories[predicted_idx]
                confidence = probs[predicted_idx].item()
                all_scores = {categories[i]: round(probs[i].item(), 4) for i in range(len(categories))}

                total += 1

                if predicted_cat != current_cat:
                    misclassified.append({
                        "file": str(img_path),
                        "current_folder": folder_name,
                        "current_category": current_cat,
                        "predicted_category": predicted_cat,
                        "confidence": round(confidence, 4),
                        "all_scores": all_scores
                    })
                    stats[current_cat]["wrong"] += 1
                else:
                    correct += 1
                    stats[current_cat]["correct"] += 1

                if confidence < confidence_threshold:
                    low_confidence.append({
                        "file": str(img_path),
                        "current_folder": folder_name,
                        "current_category": current_cat,
                        "predicted_category": predicted_cat,
                        "confidence": round(confidence, 4),
                        "all_scores": all_scores
                    })
                    stats[current_cat]["low_conf"] += 1

            except Exception as e:
                print(f"\nError processing {img_path.name}: {e}")
                total += 1

    # Print report
    print("\n" + "="*60)
    print("VERIFICATION REPORT")
    print("="*60)
    print(f"\nTotal images checked : {total}")
    print(f"Correctly classified : {correct} ({100*correct/total:.1f}%)")
    print(f"Misclassified        : {len(misclassified)} ({100*len(misclassified)/total:.1f}%)")
    print(f"Low confidence(<50%) : {len(low_confidence)}")

    print("\n--- Per Category Stats ---")
    for folder_name in ["oil_stain", "dirt_stain", "grease_stain"]:
        cat = folder_to_cat[folder_name]
        s = stats[cat]
        cat_total = s["correct"] + s["wrong"]
        pct = 100 * s["correct"] / cat_total if cat_total > 0 else 0
        print(f"  {folder_name:15s}: {s['correct']}/{cat_total} correct ({pct:.1f}%), {s['wrong']} wrong, {s['low_conf']} low-conf")

    # Save reports
    report = {
        "summary": {
            "total": total,
            "correct": correct,
            "misclassified": len(misclassified),
            "low_confidence": len(low_confidence),
            "accuracy_pct": round(100*correct/total, 2)
        },
        "misclassified_images": misclassified,
        "low_confidence_images": low_confidence
    }
    report_path = base_dir / "verification_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nFull report saved to: {report_path}")

    # Auto-fix misclassifications
    if misclassified:
        print(f"\n--- Auto-fixing {len(misclassified)} misclassified images ---")
        fixed = 0
        for item in misclassified:
            src = Path(item["file"])
            dest_folder = separated_dir / folder_names[item["predicted_category"]]
            dest = dest_folder / src.name
            counter = 1
            while dest.exists():
                dest = dest_folder / f"{src.stem}_{counter}{src.suffix}"
                counter += 1
            shutil.move(str(src), str(dest))
            fixed += 1
        print(f"Moved {fixed} images to their correct folders.")
    else:
        print("\nAll images are correctly classified! No fixes needed.")

    print("\nVerification complete!")

if __name__ == "__main__":
    main()
