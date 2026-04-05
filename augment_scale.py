import os
import cv2
import glob
import random
import numpy as np

# Settings
INPUT_FOLDER = r"C:\Users\iabhi\OneDrive\Desktop\CLUSTER_LBP_SIFT\fingerprints"
OUTPUT_FOLDER = r"C:\Users\iabhi\OneDrive\Desktop\CLUSTER_LBP_SIFT\fingerprints_scaled"
SCALE_FACTORS = [0.25, 0.50, 0.75, 1.25, 1.50, 1.75]

def augment_scale():
    print("--- 📏 Starting Scale Augmentation ---")
    
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    files = glob.glob(os.path.join(INPUT_FOLDER, "**", "*.tif"), recursive=True)
    print(f"Found {len(files)} images to process.")

    for fpath in files:
        # Maintain subdirectory structure (e.g., DB1_B)
        rel_path = os.path.relpath(fpath, INPUT_FOLDER)
        subdir = os.path.dirname(rel_path)
        output_dir = os.path.join(OUTPUT_FOLDER, subdir)
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Read Image
        img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        # Pick Random Scale
        scale = random.choice(SCALE_FACTORS)
        
        # Calculate new dimensions
        h, w = img.shape[:2]
        new_w, new_h = int(w * scale), int(h * scale)
        
        # Resize
        # INTER_AREA for shrinking, INTER_CUBIC for enlarging
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
        scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)

        # Construct new filename with metadata: name_scale_0.5.tif
        fname = os.path.basename(fpath)
        name, ext = os.path.splitext(fname)
        new_fname = f"{name}_scale_{scale}{ext}"
        
        save_path = os.path.join(output_dir, new_fname)
        cv2.imwrite(save_path, scaled_img)
        
    print(f"✅ scaling complete. Images saved to {OUTPUT_FOLDER}/")

if __name__ == "__main__":
    augment_scale()