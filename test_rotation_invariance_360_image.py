import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from main import load_models, run_hybrid_search

# UPDATE THIS PATH if needed
TEST_FOLDER = r"C:\Users\iabhi\OneDrive\Desktop\CLUSTER_LBP_SIFT\DB1_B_101_1_360Test"

def get_person_id(filename):
    """
    Extracts ID from filename.
    Example: '101_1.tif' -> '101'
    Example: '101_1_angle_005.tif' -> '101'
    """
    # Just take the first part before the first underscore
    return filename.split('_')[0]

def test_rotation_invariance_360():
    print("--- 📊 Testing 360° Rotation Invariance ---")
    
    pca_model, centroids, clustered_data = load_models()
    if not pca_model: return

    files = glob.glob(os.path.join(TEST_FOLDER, "**", "*.tif"), recursive=True)
    
    if not files:
        print(f"❌ Error: No files found in {TEST_FOLDER}")
        return

    raw_results = []
    print(f"Testing {len(files)} rotated images...")
    
    count_processed = 0

    for i, fpath in enumerate(files):
        fname = os.path.basename(fpath)
        
        # --- FIX: PARSING LOGIC FOR 360 TEST SET ---
        # Filename format: "101_1_angle_005.tif"
        try:
            if "_angle_" in fname:
                parts = fname.split("_angle_")
                angle_part = parts[1] # "005.tif"
                angle_val = int(angle_part.replace(".tif", ""))
            elif "_rot_" in fname:
                # Fallback for the other random dataset if you mix them
                parts = fname.split("_rot_")
                angle_part = parts[1]
                angle_val = int(float(angle_part.replace(".tif", "")))
            else:
                # Skip files that don't match the pattern (like the original)
                continue
        except Exception as e:
            print(f"Skipping {fname}: {e}")
            continue

        # Now we actually run the search!
        count_processed += 1
        matches = run_hybrid_search(fpath, pca_model, centroids, clustered_data)
        
        is_correct = False
        if matches:
            top_match_name = os.path.basename(matches[0][1])
            
            query_id = get_person_id(fname)
            match_id = get_person_id(top_match_name)
            
            if query_id == match_id:
                is_correct = True
            else:
                # Optional: Print wrong matches to debug
                # print(f"❌ {fname} -> {top_match_name}")
                pass
        
        raw_results.append((abs(angle_val), is_correct))
        
        if count_processed % 5 == 0:
            print(f"Processed {count_processed} images...")

    if count_processed == 0:
        print("❌ Error: Script finished but processed 0 images. Check filename format.")
        return

    # Group into buckets (0-360)
    # We create bins of 30 degrees
    bins = range(0, 361, 30) # 0, 30, 60 ... 360
    bin_labels = []
    bin_accuracies = []

    print("\n--- Results ---")
    for k in range(len(bins)-1):
        low, high = bins[k], bins[k+1]
        
        # Filter for angles in this range
        subset = [res for angle, res in raw_results if low <= angle < high]
        
        if subset:
            acc = (sum(subset) / len(subset)) * 100
            label = f"{low}-{high}°"
            bin_labels.append(label)
            bin_accuracies.append(acc)
            print(f"{label} : {acc:.1f}% ({sum(subset)}/{len(subset)})")
        else:
            bin_labels.append(f"{low}-{high}°")
            bin_accuracies.append(0)

    # Plot
    plt.figure(figsize=(12, 6))
    plt.bar(bin_labels, bin_accuracies, color='purple', alpha=0.7, edgecolor='black')
    plt.title(f"360° Rotation Invariance Accuracy\nFolder: {os.path.basename(TEST_FOLDER)}")
    plt.xlabel("Rotation Angle")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 105)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    for i, v in enumerate(bin_accuracies):
        plt.text(i, v + 1, f"{v:.0f}%", ha='center')

    plt.tight_layout()
    plt.savefig("graph_rotation_360_accuracy.png")
    plt.show()

if __name__ == "__main__":
    test_rotation_invariance_360()