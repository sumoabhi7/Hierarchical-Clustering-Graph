import os
import cv2
import glob
import random
import numpy as np
from preprocessing import rotate_image_lossless

# Settings
INPUT_FOLDER = r"C:\Users\iabhi\OneDrive\Desktop\CLUSTER_LBP_SIFT\fingerprints"
OUTPUT_FOLDER = r"C:\Users\iabhi\OneDrive\Desktop\CLUSTER_LBP_SIFT\fingerprints_rotated"

def get_random_odd_angle():
    """Generates an angle between -180 and 180, excluding +/- 10 degrees around 90s."""
    while True:
        angle = random.uniform(-180, 180)
        # Avoid multiples of 90 (within a 10 degree margin)
        if (abs(angle) < 10) or \
           (80 < abs(angle) < 100) or \
           (170 < abs(angle) <= 180):
            continue
        return int(angle)

def augment_rotate():
    print("--- 🔄 Starting Rotation Augmentation ---")
    
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    files = glob.glob(os.path.join(INPUT_FOLDER, "**", "*.tif"), recursive=True)
    print(f"Found {len(files)} images to process.")

    for fpath in files:
        rel_path = os.path.relpath(fpath, INPUT_FOLDER)
        subdir = os.path.dirname(rel_path)
        output_dir = os.path.join(OUTPUT_FOLDER, subdir)
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
        if img is None: continue

        angle = get_random_odd_angle()
        
        # Use your project's rotation function
        rotated_img = rotate_image_lossless(img, angle)

        # Save as name_rot_-45.tif
        fname = os.path.basename(fpath)
        name, ext = os.path.splitext(fname)
        new_fname = f"{name}_rot_{angle}{ext}"
        
        save_path = os.path.join(output_dir, new_fname)
        cv2.imwrite(save_path, rotated_img)

    print(f"✅ Rotation complete. Images saved to {OUTPUT_FOLDER}/")

if __name__ == "__main__":
    augment_rotate()