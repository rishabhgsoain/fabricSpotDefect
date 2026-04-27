import os
import shutil
import torch
from pathlib import Path
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

def main():
    base_dir = Path("/Users/rishabhgosain/Downloads/FabricSpotDefect An Annotated Dataset for Identifying Spot  Defects in Different Fabric Types/FabricSpotDefect")
    splits = ["train", "test", "valid"]
    output_dir = base_dir / "SeparatedDataset"
    categories = ["oil stain", "dirt stain", "grease stain"]
    
    print("Setting up zero-shot model (CLIP)...")
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")
    
    modelName = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(modelName).to(device)
    processor = CLIPProcessor.from_pretrained(modelName)
    
    text_inputs = processor(text=[f"a close up photo of fabric with an {c}" if c.startswith('oil') else f"a close up photo of fabric with a {c}" for c in categories], return_tensors="pt", padding=True).to(device)
    
    print("Starting classification & separation...")
    moved_count = {c: 0 for c in categories}
    for cat in categories:
        (output_dir / cat.replace(" ", "_")).mkdir(parents=True, exist_ok=True)
    
    image_paths = []
    for split in splits:
        split_dir = base_dir / "Orginal" / split
        if not split_dir.exists():
            continue
        for f in split_dir.iterdir():
            if f.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                image_paths.append(f)
                
    print(f"Found {len(image_paths)} images to classify.")
    
    batch_size = 1 # Using batch of 1 to keep script simple across missing corrupt items
    for img_path in tqdm(image_paths, desc="Classifying Images"):
        try:
            image = Image.open(img_path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to(device)
            inputs['input_ids'] = text_inputs['input_ids']
            inputs['attention_mask'] = text_inputs['attention_mask']
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            probs = outputs.logits_per_image.softmax(dim=1)
            predicted_idx = probs.argmax().item()
            predicted_cat = categories[predicted_idx]
            
            # Copy to output
            out_cat_dir = output_dir / predicted_cat.replace(" ", "_")
            out_img_path = out_cat_dir / img_path.name
            
            counter = 1
            while out_img_path.exists():
                out_img_path = out_cat_dir / f"{img_path.stem}_{counter}{img_path.suffix}"
                counter += 1
                
            shutil.copy2(img_path, out_img_path)
            moved_count[predicted_cat] += 1
            
        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")
            
    print("\nSeparation Complete! Image counts per category:")
    for cat, count in moved_count.items():
        print(f"  {cat}: {count} images")
        
    print(f"\nAll separated images are in: {output_dir}")

if __name__ == "__main__":
    main()
