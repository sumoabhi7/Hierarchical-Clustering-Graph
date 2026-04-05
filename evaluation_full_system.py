import os
import time
import numpy as np
from main import load_models, run_hybrid_search

# --- SETTINGS ---
TOP_N_TO_CHECK = 3  # We check if a valid match is in the top N results

def get_person_id(filename):
    """
    Extracts ID from filename.
    Format expected: '101_1.tif' -> '101'
    """
    base = os.path.basename(filename)
    # Split by '_' and take the first part (Person ID)
    return base.split('_')[0]

def evaluate_accuracy():
    print(f"--- 🧪 Starting Full Database Accuracy Test (Rank-{TOP_N_TO_CHECK}) ---")
    
    # 1. Load System Models
    pca_model, centroids, clustered_data = load_models()
    if not pca_model:
        print("❌ Error: Models not found.")
        return

    # Get all images currently in the DB
    all_files = clustered_data["fnames"]
    total_images = len(all_files)
    
    if total_images == 0:
        print("❌ Database is empty.")
        return

    print(f"📂 Found {total_images} fingerprints in the database.")
    print("🚀 Running search for every image... (This may take some time)")
    print("-" * 60)

    correct_count = 0
    start_time_all = time.time()

    # 2. Iterate through every image
    for i, query_path in enumerate(all_files):
        query_filename = os.path.basename(query_path)
        query_id = get_person_id(query_filename)
        
        # Run the search
        # We silence the print statements inside run_hybrid_search by not printing them here, 
        # but the function itself prints. To reduce clutter, you might want to comment out prints in main.py,
        # but for now we just run it.
        matches = run_hybrid_search(query_path, pca_model, centroids, clustered_data)
        
        # 3. Check for Valid Match (excluding self)
        found_valid_match = False
        
        # Check matching results
        if matches:
            # We look at the top N matches
            for rank, (score, match_path) in enumerate(matches[:TOP_N_TO_CHECK]):
                match_filename = os.path.basename(match_path)
                
                # IGNORE SELF-MATCH (Trivial)
                if match_filename == query_filename:
                    continue
                
                # Check ID
                match_id = get_person_id(match_filename)
                
                if match_id == query_id:
                    found_valid_match = True
                    break # Found a correct sibling!
        
        if found_valid_match:
            correct_count += 1
            status = "✅ PASS"
        else:
            status = "❌ FAIL"
            
        # Optional: Print progress line
        # print(f"[{i+1}/{total_images}] {query_filename} -> {status}")

        # Progress Indicator every 10 images
        if (i + 1) % 10 == 0:
            print(f"Processed {i+1}/{total_images} ... Current Acc: {(correct_count/(i+1))*100:.1f}%")

    total_time = time.time() - start_time_all
    
    # 4. Final Report
    final_accuracy = (correct_count / total_images) * 100
    
    print("-" * 60)
    print("--- 📊 FINAL RESULTS ---")
    print(f"Total Images Tested: {total_images}")
    print(f"Successful Identifications: {correct_count}")
    print(f"Failed Identifications: {total_images - correct_count}")
    print(f"Time Taken: {total_time:.1f} seconds ({total_time/total_images:.2f}s per query)")
    print("-" * 30)
    print(f"🏆 SYSTEM ACCURACY (Rank-{TOP_N_TO_CHECK}): {final_accuracy:.2f}%")
    print("-" * 30)

if __name__ == "__main__":
    evaluate_accuracy()