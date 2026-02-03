# build_database.py
import os
import cv2
import numpy as np
import pickle
import glob
from preprocessing import enhance_fingerprint
from feature_extraction import extract_lbp_features_for_clustering, extract_sift_features, serialize_keypoints
from pca_reduction import train_pca
from clustering import perform_hierarchical_clustering, calculate_centroids

DB_FOLDER = "fingerprints" 
OUTPUT_DB = "clustered_data.pkl"

def build_database():
    print("Building database with SIFT + LBP Clustering...")
    
    # 1. Gather files
    files = glob.glob(os.path.join(DB_FOLDER, "**", "*.tif"), recursive=True)
    if not files:
        print("No .tif files found in", DB_FOLDER)
        return

    print(f"Found {len(files)} images.")

    fnames = []
    lbp_features_list = [] # For clustering
    sift_db = {}           # For matching

    # 2. Extract Features
    for fpath in files:
        try:
            # We use the enhanced image for both
            enhanced = enhance_fingerprint(fpath, return_binarized=False)
            if enhanced is None: continue

            # A. Extract LBP (for Clustering)
            lbp_desc = extract_lbp_features_for_clustering(enhanced)
            
            # B. Extract SIFT (for Matching)
            kp, sift_des = extract_sift_features(enhanced)

            if lbp_desc is not None and sift_des is not None and kp is not None:
                fnames.append(fpath)
                lbp_features_list.append(lbp_desc)
                
                # Store SIFT descriptors AND serialized keypoints in the DB
                sift_db[fpath] = {
                    "sift_des": sift_des,
                    "sift_kp": serialize_keypoints(kp)
                }
                
                if len(fnames) % 10 == 0:
                    print(f"Processed {len(fnames)}...")
        except Exception as e:
            print(f"Skipping {fpath}: {e}")

    if not fnames:
        print("No valid features extracted.")
        return

    # 3. Train PCA & Cluster (On LBP features)
    print("Training PCA...")
    pca_model, reduced_features = train_pca(np.array(lbp_features_list), n_components=0.95)
    with open("pca_model.pkl", "wb") as f:
        pickle.dump(pca_model, f)

    print("Clustering...")
    model, labels = perform_hierarchical_clustering(reduced_features)
    centroids = calculate_centroids(reduced_features, labels)
    with open("cluster_centroids.pkl", "wb") as f:
        pickle.dump(centroids, f)

    # 4. Save Final Database
    print("Saving Database...")
    final_data = {
        "fnames": fnames,
        "cluster_labels": labels,
        "sift_database": sift_db 
    }

    with open(OUTPUT_DB, "wb") as f:
        pickle.dump(final_data, f)
    
    print("Database build complete!")

if __name__ == "__main__":
    build_database()