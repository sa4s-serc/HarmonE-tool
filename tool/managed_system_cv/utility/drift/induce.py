#!/usr/bin/env python3
"""
induce_drift.py

Applies artificial drift to BDD100K train images using statistically
isolated methods.
- 'dark' reduces mean luminance while preserving standard deviation.
- 'fog' reduces standard deviation while preserving mean luminance.

The script splits the dataset into 7 parts of RANDOMIZED sizes that are
guaranteed to sum to a TARGET_SUM (70,000).

The use of a fixed random seed ensures the randomized intervals are
reproducible for experimental consistency.
"""

import os
import random
import shutil
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision.transforms.functional import adjust_brightness

# =======================
# CONFIG
# =======================
INPUT_DIR = "data/bdd100k/images/train_unchanged"
OUTPUT_DIR = "data/bdd100k/images/train"
TOTAL_PARTS = 7
TARGET_SUM = 70000
RANDOM_SEED = 1

# Part size will be randomized within this range
MIN_PART_SIZE = 8000
MAX_PART_SIZE = 12000
PART_SIZE_STEP = 500

# --- NEW: Statistically-driven augmentation factors ---
# For 'dark', we shift the mean luminance to this target value.
LUMINANCE_FACTOR = 0.35
# For 'fog', we scale the standard deviation by this factor.
CONTRAST_FACTOR = 0.25

# =======================
# UTILS
# =======================
def load_image_paths(folder):
    """Loads and sorts all image paths from a directory."""
    exts = (".jpg", ".jpeg", ".png")
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(exts)]
    files.sort()
    return files

# --- NEW: Statistically Pure Augmentation Functions ---

def apply_darkness(img, factor):
    return adjust_brightness(img, factor)

def apply_fog(image_array, factor):
    """
    Reduces the contrast (standard deviation) of an image by a factor,
    while preserving the mean.
    """
    # Calculate the mean for each color channel independently
    mean = np.mean(image_array, axis=(0, 1), keepdims=True)
    
    # Apply the contrast formula and clip to the valid [0, 255] range
    foggy_array = np.clip((image_array - mean) * factor + mean, 0, 255)
    return foggy_array

# --- (This function is still needed for generating part sizes) ---
def generate_constrained_partitions(num_parts, total_sum, min_val, max_val, step):
    """
    Generates a list of random integers that sum to a target value,
    with each integer respecting min/max/step constraints.
    """
    random.seed(RANDOM_SEED)
    partitions = []
    remaining_sum = total_sum

    for i in range(num_parts - 1):
        parts_left_to_generate = num_parts - (i + 1)
        lower_bound = max(min_val, remaining_sum - (parts_left_to_generate * max_val))
        upper_bound = min(max_val, remaining_sum - (parts_left_to_generate * min_val))
        lower_bound = int(np.ceil(lower_bound / step) * step)
        upper_bound = int(np.floor(upper_bound / step) * step)

        if lower_bound > upper_bound:
            raise ValueError("Cannot create a valid partition with the given constraints.")

        part_size = random.randrange(lower_bound, upper_bound + 1, step)
        partitions.append(part_size)
        remaining_sum -= part_size

    partitions.append(remaining_sum)
    return partitions

# =======================
# MAIN
# =======================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    images = load_image_paths(INPUT_DIR)
    n_images = len(images)
    print(f"Found {n_images} images in {INPUT_DIR}")

    if n_images < TARGET_SUM:
        print(f"Warning: Number of images ({n_images}) is less than the target sum ({TARGET_SUM}).")

    aug_sequence = ["clear", "dark", "clear", "fog", "dark", "clear", "fog"]

    try:
        part_sizes = generate_constrained_partitions(
            TOTAL_PARTS, TARGET_SUM, MIN_PART_SIZE, MAX_PART_SIZE, PART_SIZE_STEP
        )
        assert sum(part_sizes) == TARGET_SUM
        print(f"Generated constrained random part sizes (reproducible): {part_sizes}")
    except ValueError as e:
        print(f"Error: {e}. Could not generate valid partitions. Check constraints.")
        return

    current_idx = 0
    for part_idx, (aug_type, part_size) in enumerate(zip(aug_sequence, part_sizes)):
        if current_idx >= n_images:
            break

        start = current_idx
        end   = min(start + part_size, n_images)
        part_imgs = images[start:end]

        print(f"\nProcessing Part {part_idx+1}/{TOTAL_PARTS} [{aug_type}] with {len(part_imgs)} images (from index {start} to {end-1})")

        for path in tqdm(part_imgs, desc=f"Part {part_idx+1} - {aug_type}"):
            rel_path = os.path.relpath(path, INPUT_DIR)
            out_path = os.path.join(OUTPUT_DIR, rel_path)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)

            if aug_type == "clear":
                shutil.copy(path, out_path)
                continue

            try:
                with Image.open(path).convert("RGB") as img:
                    # Convert to NumPy array for statistical manipulation
                    img_array = np.array(img, dtype=np.float32)
                    
                    if aug_type == "dark":
                        aug = apply_darkness(img, LUMINANCE_FACTOR)
                        aug_array = np.array(aug, dtype=np.float32)
                    elif aug_type == "fog":
                        aug_array = apply_fog(img_array, CONTRAST_FACTOR)
                    
                    # Convert back to PIL Image and save
                    aug_img = Image.fromarray(aug_array.astype(np.uint8))
                    aug_img.save(out_path)

            except Exception as e:
                print(f"Warning: Failed to process {path}. Error: {e}")
                continue
        
        current_idx = end

    print(f"\n✅ Randomized drift induction complete. Processed {current_idx} images.")
    print(f"Augmented dataset written to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()