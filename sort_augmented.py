"""
Sort Augmented Images by Stain Type
====================================
Instead of running CLIP on all 1,929 augmented images again (slow),
we match each augmented image to its original using the filename prefix,
then copy the original's label. This runs in seconds.

Augmented filenames follow the pattern:
  <original_base>_jpg.rf.<new_hash>.jpg

So "20240204_085317_jpg.rf.NEWHASH.jpg" was augmented from
   "20240204_085317_jpg.rf.OLDHASH.jpg"

We extract the prefix up to "_jpg.rf." and use it as the key.
"""

import os
import shutil
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

BASE_DIR = Path("/Users/rishabhgosain/Downloads/FabricSpotDefect An Annotated Dataset for Identifying Spot  Defects in Different Fabric Types/FabricSpotDefect")
SEPARATED_DIR = BASE_DIR / "SeparatedDataset"
AUG_SOURCE_DIR = BASE_DIR / "Augmented" / "COCO" / "train"
AUG_OUTPUT_DIR = SEPARATED_DIR  # Put augmented images in same 3 folders


def get_key(filename):
    """Extract the time-based prefix before the '_jpg.rf.' hash suffix."""
    name = Path(filename).stem  # remove .jpg
    if "_jpg.rf." in name:
        return name.split("_jpg.rf.")[0]
    return name  # fallback: use whole stem


def build_original_label_map():
    """Build a dict: prefix_key -> stain_label from already-sorted originals."""
    label_map = {}
    categories = ["oil_stain", "dirt_stain", "grease_stain"]
    for cat in categories:
        cat_dir = SEPARATED_DIR / cat
        if not cat_dir.exists():
            continue
        for f in cat_dir.iterdir():
            if f.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                key = get_key(f.name)
                label_map[key] = cat
    return label_map


def main():
    print("Building label map from sorted originals...")
    label_map = build_original_label_map()
    print(f"  Found {len(label_map)} unique original image keys.\n")

    # Collect all augmented images
    aug_images = [f for f in AUG_SOURCE_DIR.iterdir()
                  if f.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    print(f"Augmented images to sort: {len(aug_images)}\n")

    counts = defaultdict(int)
    not_found = []

    for img_path in tqdm(aug_images, desc="Sorting augmented images"):
        key = get_key(img_path.name)
        label = label_map.get(key)

        if label is None:
            not_found.append(img_path.name)
            continue

        dest_dir = AUG_OUTPUT_DIR / label
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / img_path.name

        # Avoid overwriting if already exists from original sort
        if dest.exists():
            stem = img_path.stem
            suffix = img_path.suffix
            dest = dest_dir / f"{stem}_aug{suffix}"

        shutil.copy2(img_path, dest)
        counts[label] += 1

    print("\n=== Done! Augmented Images Added ===")
    for cat, n in sorted(counts.items()):
        print(f"  {cat}: +{n} images")

    if not_found:
        print(f"\n  Could not match {len(not_found)} images (no key in original set) — skipped.")

    print("\nNow re-run train_classifier.py to train on the expanded dataset.")


if __name__ == "__main__":
    main()
