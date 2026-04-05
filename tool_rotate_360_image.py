import os
import cv2
import numpy as np
from preprocessing import rotate_image_lossless

# --- CONFIGURATION ---
# Path to the specific image you want to test
TARGET_IMAGE_PATH = "fingerprints/DB1_B/101_1.tif" 

# Rotation Step (5 degrees)
STEP_ANGLE = 5

def generate_360_dataset():
    print(f"--- 🔄 Generatng 360° Test Set for: {TARGET_IMAGE_PATH} ---")

    # 1. Validation
    if not os.path.exists(TARGET_IMAGE_PATH):
        print(f"❌ Error: File not found at {TARGET_IMAGE_PATH}")
        return

    # 2. Parse Names for Output Folder
    # Example: fingerprints/DB1_B/101_1.tif
    # Parent Dir: DB1_B
    # File Name: 101_1
    parent_dir_name = os.path.basename(os.path.dirname(TARGET_IMAGE_PATH))
    file_name_no_ext = os.path.splitext(os.path.basename(TARGET_IMAGE_PATH))[0]
    
    # Output Folder Name: "DB1_B_101_1_360Test"
    output_folder = f"{parent_dir_name}_{file_name_no_ext}_360Test"
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"📁 Created output folder: {output_folder}")
    else:
        print(f"📁 Using existing folder: {output_folder}")

    # 3. Load Image
    img = cv2.imread(TARGET_IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("❌ Error: Could not read valid image data.")
        return

    # 4. Generate Rotations
    count = 0
    # Range(0, 361, 5) includes 0, 5, 10 ... up to 360
    for angle in range(0, 361, STEP_ANGLE):
        
        # Use your project's lossless rotation (Handles white padding/background)
        # This prevents the "straight line" artifacts
        rotated_img = rotate_image_lossless(img, angle)
        
        # Construct Filename: 101_1_angle_005.tif
        # We use :03d to pad zeros (e.g. 005, 010) for easier sorting
        save_name = f"{file_name_no_ext}_angle_{angle:03d}.tif"
        save_path = os.path.join(output_folder, save_name)
        
        cv2.imwrite(save_path, rotated_img)
        count += 1
    
    print("-" * 30)
    print(f"✅ Success! Generated {count} images.")
    print(f"📂 Location: {os.path.abspath(output_folder)}")

if __name__ == "__main__":
    generate_360_dataset()