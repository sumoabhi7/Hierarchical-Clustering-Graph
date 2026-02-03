import os
import glob
import matplotlib.pyplot as plt
from main import load_models, run_hybrid_search

TEST_FOLDER = "fingerprints_scaled"

def get_person_id(filename):
    """
    Extracts ID from filename.
    Example: '101_1.tif' -> '101'
    Example: '101_1_scale_0.5.tif' -> '101'
    """
    # Split by underscore and take the first part
    return filename.split('_')[0]

def test_scale_invariance():
    print("--- 📊 Testing Scale Invariance (ID Matching) ---")
    
    pca_model, centroids, clustered_data = load_models()
    if not pca_model: return

    files = glob.glob(os.path.join(TEST_FOLDER, "**", "*.tif"), recursive=True)
    
    # Store results: { 0.25: [True, False...], ... }
    results_by_scale = {}

    print(f"Testing {len(files)} scaled images...")
    
    for i, fpath in enumerate(files):
        fname = os.path.basename(fpath)
        
        # Parse Scale Value from: "101_1_scale_0.5.tif"
        try:
            # Split by '_scale_' to find the value
            parts = fname.split("_scale_")
            scale_part = parts[1] # "0.5.tif"
            scale_val = float(scale_part.replace(".tif", ""))
        except:
            continue

        if scale_val not in results_by_scale:
            results_by_scale[scale_val] = []

        # Run Search
        matches = run_hybrid_search(fpath, pca_model, centroids, clustered_data)
        
        is_correct = False
        if matches:
            # Get the filename of the Top Match (Rank 1)
            top_match_path = matches[0][1]
            top_match_name = os.path.basename(top_match_path)
            
            # Compare Person IDs
            query_id = get_person_id(fname)          # '101' from query
            match_id = get_person_id(top_match_name) # '101' from DB result
            
            if query_id == match_id:
                is_correct = True
        
        results_by_scale[scale_val].append(is_correct)
        
        if i % 10 == 0:
            print(f"Processed {i}/{len(files)}...")

    # Calculate Accuracy
    scales = sorted(results_by_scale.keys())
    accuracies = []
    
    print("\n--- Correct Identity Match Results ---")
    for s in scales:
        total = len(results_by_scale[s])
        correct = sum(results_by_scale[s])
        acc = (correct / total) * 100
        accuracies.append(acc)
        print(f"Scale {s}x : {acc:.1f}% ({correct}/{total})")

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(scales, accuracies, marker='o', color='green', linewidth=2)
    plt.title("Scale Invariance Accuracy (Same Person ID)")
    plt.xlabel("Scale Factor")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 105)
    plt.grid(True)
    
    for x, y in zip(scales, accuracies):
        plt.text(x, y + 2, f"{y:.1f}%", ha='center')
    
    plt.savefig("graph_scale_accuracy_id.png")
    plt.show()

if __name__ == "__main__":
    test_scale_invariance()