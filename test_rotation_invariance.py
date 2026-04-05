import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from main import load_models, run_hybrid_search

# Folder containing the random rotation images
TEST_FOLDER = "fingerprints_rotated"

def get_person_id(filename):
    """
    Extracts ID from filename.
    """
    return filename.split('_')[0]

def get_db_name(filepath):
    """
    Extracts the parent folder name (Database Name) from the file path.
    Example: 'fingerprints_rotated/DB1_B/101_1.tif' -> 'DB1_B'
    """
    return os.path.basename(os.path.dirname(filepath))

def test_rotation_invariance_per_db():
    print(f"--- 📊 Testing Rotation Invariance (Per DB) on {TEST_FOLDER} ---")
    
    pca_model, centroids, clustered_data = load_models()
    if not pca_model: return

    files = glob.glob(os.path.join(TEST_FOLDER, "**", "*.tif"), recursive=True)
    
    if not files:
        print(f"❌ Error: No files found in {TEST_FOLDER}")
        return

    # Dictionary to store results per DB
    # Structure: { "DB1_B": { "total": 0, "correct": 0, "raw_data": [] }, "DB2_B": ... }
    db_stats = {}

    print(f"Testing {len(files)} rotated images...")
    
    count_processed = 0

    for i, fpath in enumerate(files):
        fname = os.path.basename(fpath)
        
        # 1. Identify Database
        db_name = get_db_name(fpath)
        if db_name not in db_stats:
            db_stats[db_name] = {"total": 0, "correct": 0, "raw_data": []}

        # 2. Parse Angle
        try:
            if "_angle_" in fname:
                parts = fname.split("_angle_")
                angle_val = int(parts[1].replace(".tif", ""))
            elif "_rot_" in fname:
                parts = fname.split("_rot_")
                angle_val = int(float(parts[1].replace(".tif", "")))
            else:
                continue
        except Exception:
            continue

        # 3. Run Search
        count_processed += 1
        matches = run_hybrid_search(fpath, pca_model, centroids, clustered_data)
        
        is_correct = False
        if matches:
            top_match_name = os.path.basename(matches[0][1])
            if get_person_id(fname) == get_person_id(top_match_name):
                is_correct = True
        
        # 4. Store Data
        db_stats[db_name]["total"] += 1
        if is_correct:
            db_stats[db_name]["correct"] += 1
        
        # Store angle data for this specific DB (for potentially plotting later)
        db_stats[db_name]["raw_data"].append((abs(angle_val), is_correct))
        
        if count_processed % 10 == 0:
            print(f"Processed {count_processed}...")

    if count_processed == 0:
        print("❌ Error: Processed 0 images.")
        return

    # --- PRINT RESULTS ---
    print("\n" + "="*40)
    print(f"{'DATABASE':<15} | {'ACCURACY':<10} | {'CORRECT/TOTAL'}")
    print("-" * 40)
    
    sorted_dbs = sorted(db_stats.keys())
    
    # Lists for plotting
    plot_names = []
    plot_accs = []

    for db in sorted_dbs:
        stats = db_stats[db]
        if stats["total"] > 0:
            acc = (stats["correct"] / stats["total"]) * 100
            print(f"{db:<15} | {acc:>6.1f}%    | {stats['correct']}/{stats['total']}")
            plot_names.append(db)
            plot_accs.append(acc)
        else:
            print(f"{db:<15} | {'N/A':>6}    | 0/0")

    print("="*40)

    # --- PLOT BAR CHART ---
    if plot_names:
        plt.figure(figsize=(10, 6))
        colors = plt.cm.viridis(np.linspace(0, 0.8, len(plot_names))) # Nice colors
        bars = plt.bar(plot_names, plot_accs, color=colors, edgecolor='black', alpha=0.8)
        
        plt.title("Rotation Invariance Accuracy by Database")
        plt.xlabel("Database Name")
        plt.ylabel("Accuracy (%)")
        plt.ylim(0, 105)
        plt.grid(axis='y', linestyle='--', alpha=0.5)

        # Add percentage labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                     f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        plt.savefig("graph_accuracy_per_db.png")
        plt.show()

if __name__ == "__main__":
    test_rotation_invariance_per_db()