"""
Relabel YOLOv8 Dataset for Multi-Class Stain Detection

This script converts existing YOLO `.txt` label files which currently use
a generic single class '0' (spot) into multi-class labels (0=dirt, 1=grease, 2=oil)
based on the classification map from the SeparatedDataset.
"""

import os
from pathlib import Path
from tqdm import tqdm

BASE_DIR = Path("/Users/rishabhgosain/Downloads/FabricSpotDefect An Annotated Dataset for Identifying Spot  Defects in Different Fabric Types/FabricSpotDefect")
SEPARATED_DIR = BASE_DIR / "SeparatedDataset"

# Identify the YOLO directories we want to update
YOLO_DIRS = [
    BASE_DIR / "Augmented" / "YOLOv8",
    BASE_DIR / "Original" / "YOLOv8"
]

CLASS_IDS = {
    "dirt_stain": 0,
    "grease_stain": 1,
    "oil_stain": 2
}

def get_key(filename):
    """Extract the time-based prefix before the '_jpg.rf.' hash suffix."""
    name = Path(filename).stem  # remove .txt or .jpg
    if "_jpg.rf." in name:
        return name.split("_jpg.rf.")[0]
    return name

def build_label_map():
    """Build a mapping of image_key -> new_class_id based on separated folders."""
    label_map = {}
    for cat, class_id in CLASS_IDS.items():
        cat_dir = SEPARATED_DIR / cat
        if not cat_dir.exists():
            continue
        for f in cat_dir.iterdir():
            if f.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                key = get_key(f.name)
                label_map[key] = class_id
    return label_map

def process_labels(yolo_dir, label_map):
    labels_changed = 0
    not_found = 0
    total_boxes = 0
    
    if not yolo_dir.exists():
        return 0, 0, 0
    
    # Process train, val, test splits
    for split in ["train", "valid", "test"]:
        labels_dir = yolo_dir / split / "labels"
        if not labels_dir.exists():
            continue
            
        for txt_file in tqdm(list(labels_dir.glob("*.txt")), desc=f"Processing {yolo_dir.name}/{split}"):
            key = get_key(txt_file.name)
            new_class_id = label_map.get(key)
            
            if new_class_id is None:
                not_found += 1
                continue
                
            # Read existing lines
            lines = txt_file.read_text().strip().split("\n")
            if not lines or lines == ['']:
                continue
                
            new_lines = []
            file_changed = False
            for line in lines:
                parts = line.split()
                if not parts:
                    continue
                # YOLO format: class_id x_center y_center width height
                # Overwrite class_id with new_class_id
                old_class_id = parts[0]
                if int(old_class_id) != new_class_id:
                    file_changed = True
                
                new_line = f"{new_class_id} " + " ".join(parts[1:])
                new_lines.append(new_line)
                total_boxes += 1
                
            # Write back if we mapped it
            if file_changed:
                txt_file.write_text("\n".join(new_lines) + "\n")
                labels_changed += 1
                
    return labels_changed, not_found, total_boxes

def main():
    print("Building label map from SeparatedDataset...")
    label_map = build_label_map()
    print(f"Loaded {len(label_map)} labeled image keys.\n")
    
    total_changed = 0
    total_missing = 0
    total_box_count = 0
    
    for ydir in YOLO_DIRS:
        changed, missing, boxes = process_labels(ydir, label_map)
        total_changed += changed
        total_missing += missing
        total_box_count += boxes
        
    print("\n--- Relabeling Complete ---")
    print(f"Files updated with multi-class IDs: {total_changed}")
    print(f"Total bounding boxes updated: {total_box_count}")
    print(f"Unmapped files left as-is: {total_missing}")
    
    # Generate new stain_data.yaml
    yaml_content = f"""
path: ./Augmented/YOLOv8  # relative to ultralytics runner
train: train/images
val: valid/images
test: test/images

# Classes
names:
  0: dirt_stain
  1: grease_stain
  2: oil_stain
"""
    yaml_path = BASE_DIR / "stain_data.yaml"
    yaml_path.write_text(yaml_content.strip())
    print(f"\nCreated new YOLO config: {yaml_path}")

if __name__ == "__main__":
    main()
