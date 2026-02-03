# main.py
import pickle
import numpy as np
import cv2
import time
import os

from preprocessing import enhance_fingerprint
from feature_extraction import extract_lbp_features_for_clustering, extract_sift_features, deserialize_keypoints
from identification import find_top_matches_sift
from visualizer import visualize_top_matches, select_query_image_gui
from utils import load_fingerprint_database

TOP_MATCHES_TO_RETURN = 3
# ACCURACY FIX: Increased to search more clusters. 
# This helps when rotation shifts the LBP histogram slightly.
TOP_N_CLUSTERS = 9

def load_models():
    print("Loading models...")
    try:
        with open("pca_model.pkl", "rb") as f:
            pca_model = pickle.load(f)
        with open("cluster_centroids.pkl", "rb") as f:
            centroids = pickle.load(f)
        clustered_data = load_fingerprint_database("clustered_data.pkl")
    except Exception as e:
        print("Error loading models:", e)
        return None, None, None

    print("Models loaded.")
    return pca_model, centroids, clustered_data

def run_hybrid_search(query_path, pca_model, centroids, clustered_data, sigma=0):
    start_time = time.time()
    
    # 1. Preprocessing
    enhanced = enhance_fingerprint(query_path, blur_sigma=sigma, return_binarized=False)
    if enhanced is None:
        print("Enhancement failed.")
        return []

    # 2. Extract Features
    query_lbp = extract_lbp_features_for_clustering(enhanced)
    query_kp, query_sift_des = extract_sift_features(enhanced)
    
    if query_sift_des is None:
        print("No SIFT features found in query image.")
        return []

    # 3. Cluster Filtering
    query_pca = pca_model.transform(query_lbp.reshape(1, -1))[0]
    
    labels = list(centroids.keys())
    centroid_matrix = np.stack([centroids[l] for l in labels])
    dists_to_centroids = np.sum((centroid_matrix - query_pca) ** 2, axis=1)
    
    # Select Top N Clusters
    top_label_idx = np.argsort(dists_to_centroids)[:TOP_N_CLUSTERS]
    top_labels = [labels[i] for i in top_label_idx]

    fnames_db = np.array(clustered_data["fnames"])
    cluster_labels_db = clustered_data["cluster_labels"]
    full_db = clustered_data["sift_database"] 

    mask = np.zeros_like(cluster_labels_db, dtype=bool)
    for l in top_labels:
        mask = mask | (cluster_labels_db == l)
    
    filtered_paths = fnames_db[mask]
    
    filtered_db_unpacked = {}
    for p in filtered_paths:
        if p in full_db:
            entry = full_db[p]
            filtered_db_unpacked[p] = {
                "sift_des": entry["sift_des"],
                "sift_kp": deserialize_keypoints(entry["sift_kp"]) 
            }

    print(f"Scanning {len(filtered_db_unpacked)} candidates (Top {TOP_N_CLUSTERS} clusters)...")

    # 4. SIFT + RANSAC Matching
    top_matches = find_top_matches_sift((query_kp, query_sift_des), filtered_db_unpacked, top_n=TOP_MATCHES_TO_RETURN)

    end_time = time.time()
    
    if top_matches:
        print("\n--- Top Matches ---")
        for score, path in top_matches:
            db_folder = os.path.basename(os.path.dirname(path))
            fname = os.path.basename(path)
            print(f"[{db_folder}] {fname} -> Inliers: {score}")
    else:
        print("No matches found.")

    return top_matches

if __name__ == "__main__":
    pca_model, centroids, clustered_data = load_models()
    
    if pca_model:
        while True:
            print("\n" + "-"*40)
            print("Please select a query image (or Cancel to exit)...")
            query_path = select_query_image_gui()
            
            if not query_path:
                print("Goodbye!")
                break
                
            print(f"Processing: {os.path.basename(query_path)}")
            matches = run_hybrid_search(query_path, pca_model, centroids, clustered_data)
            
            if matches:
                visualize_top_matches(query_path, matches)