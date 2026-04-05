# visualizer.py
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from scipy.cluster.hierarchy import dendrogram, linkage
from typing import List, Tuple
import tkinter as tk
from tkinter import filedialog

# --- 1. CLUSTER VISUALIZATION ---
def visualize_clusters(pca_features: np.ndarray, cluster_labels: np.ndarray, fnames: List[str]):
    if pca_features.shape[1] < 2:
        print("❌ Cannot visualize clusters: PCA features must have at least 2 dimensions.")
        return

    print("🎨 Generating 2D PCA Scatter Plot...")
    
    x = pca_features[:, 0]
    y = pca_features[:, 1]
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(x, y, c=cluster_labels, cmap='viridis', s=50, alpha=0.6)
                          
    plt.title('Fingerprint Features Clustered in PCA Space (LBP)')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    
    cbar = plt.colorbar(scatter)
    cbar.set_label('Cluster Label')
    plt.grid(True, linestyle='--', alpha=0.5)
    print("✅ Cluster plot ready.")
    plt.show() 

def visualize_dendrogram(pca_features: np.ndarray):
    print("🌲 Generating Dendrogram...")
    Z = linkage(pca_features, method='ward')
    
    plt.figure(figsize=(15, 7))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Sample Index')
    plt.ylabel('Distance (Ward Linkage)')
    
    dendrogram(
        Z,
        leaf_rotation=90.,
        leaf_font_size=8.,
        truncate_mode='lastp',
        p=20,
        show_leaf_counts=True,
    )
    plt.tight_layout()
    print("✅ Dendrogram ready.")
    plt.show()

# --- 2. MATCH RESULTS VISUALIZATION ---
def visualize_top_matches(query_path: str, top_matches: List[Tuple[float, str]]):
    """
    Creates a bar plot of match scores and displays images with DB names.
    """
    if not top_matches:
        print("❌ Cannot visualize matches: Match results list is empty.")
        return

    # 2a. Bar Plot
    print("🎨 Generating Top Matches Bar Plot...")
    scores = [m[0] for m in top_matches]
    labels = []
    
    for m in top_matches:
        path = m[1]
        # Extract DB Name (Parent Folder)
        db_name = os.path.basename(os.path.dirname(path))
        file_name = os.path.basename(path)
        # Format: [DB1_B] 101_1.tif
        label = f"[{db_name}] {file_name}"
        labels.append(label)
    
    y_pos = np.arange(len(labels))
    plt.figure(figsize=(12, 6)) # Slightly wider for long names
    
    # Use green for matches
    plt.barh(y_pos, scores, color='lightgreen', align='center')
    plt.gca().invert_yaxis()
    plt.yticks(y_pos, labels)
    plt.xlabel('Number of RANSAC Inliers (Geometric Matches)')
    plt.title(f'Top Matches for: {os.path.basename(query_path)}')
    
    for i, score in enumerate(scores):
        plt.text(score, i, f'{int(score)}', ha='left', va='center')
        
    print("✅ Match plot ready.")
    plt.show() 
    
    # 2b. Image Display
    print("\n🖼️ Displaying Search and Match Images...")
    num_images = len(top_matches) + 1
    
    plt.figure(figsize=(15, 5)) 

    _display_single_image(query_path, 1, num_images, 1, f"Query: {os.path.basename(query_path)}")
    
    for i, (score, match_path) in enumerate(top_matches):
        db_name = os.path.basename(os.path.dirname(match_path))
        file_name = os.path.basename(match_path)
        title = f"Match {i+1}\n[{db_name}]\n{file_name}\nScore: {int(score)}"
        _display_single_image(match_path, 1, num_images, i + 2, title)

    plt.tight_layout()
    print("✅ Image display ready.")
    plt.show()

def _display_single_image(image_path: str, rows: int, cols: int, index: int, title: str):
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) 
        if img is None:
            raise FileNotFoundError(f"Image not found/loaded: {image_path}")
        plt.subplot(rows, cols, index)
        plt.imshow(img, cmap='gray')
        plt.title(title, fontsize=9)
        plt.axis('off')
    except Exception as e:
        print(f"⚠️ Error displaying image {image_path}: {e}")
        plt.subplot(rows, cols, index)
        plt.text(0.5, 0.5, "Image Error", ha='center', va='center')
        plt.title(title)
        plt.axis('off')

def select_query_image_gui():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select Query Fingerprint Image",
        filetypes=[("Image Files", "*.tif *.tiff *.png *.jpg *.jpeg *.bmp")]
    )
    root.destroy()
    return file_path if file_path else None