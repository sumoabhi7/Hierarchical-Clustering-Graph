from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from scipy.cluster.hierarchy import dendrogram, linkage
import time

# -----------------------------------------------------------------
# FINAL, COMPLETE CODE FOR HIERARCHICAL ANALYSIS OF A LARGE IMAGE
# -----------------------------------------------------------------

# This script performs a multi-level hierarchical analysis on a large image
# using a scalable, hybrid approach. It generates a clustered image, a
# high-level dendrogram of the main color groups (centroids), and then
# a series of detailed dendrograms for each of those main color groups.

# -----------------------
# Step 1: Load and Preprocess the Image
# -----------------------
# Loads a large image and reshapes it into a 2D array of pixels,
# where each row is a data point to be clustered.
try:
    image_path = "blackhole.jpg"  # <-- Change this to your image path
    original_img = Image.open(image_path).convert('RGB')
    # Resize to a large size to demonstrate scalability
    large_img = original_img.resize((200, 200))
    img_array = np.array(large_img)
    pixels = img_array.reshape(-1, 3)
    print(f"Large image loaded. Shape: {img_array.shape}, Pixels: {pixels.shape}")
except FileNotFoundError:
    print("Image not found. Creating a large dummy image for demonstration.")
    large_img = np.random.randint(0, 256, (500, 500, 3), dtype=np.uint8)
    img_array = large_img
    pixels = img_array.reshape(-1, 3)

# -----------------------
# Step 2: Use MiniBatchKMeans for Scalable Top-Level Clustering
# -----------------------
# MiniBatchKMeans is used on the ENTIRE dataset to find a small number of
# top-level clusters efficiently, without the memory issues of AgglomerativeClustering.
n_top_level_clusters = 5
print(f"\nFinding {n_top_level_clusters} top-level clusters with MiniBatchKMeans...")
start_time = time.time()
kmeans_model = MiniBatchKMeans(n_clusters=n_top_level_clusters, random_state=0, n_init=10)
kmeans_model.fit(pixels)
labels = kmeans_model.predict(pixels)
centroids = kmeans_model.cluster_centers_
print(f"MiniBatchKMeans on full dataset took {time.time() - start_time:.2f} seconds.")

# -----------------------
# Step 2.1: Count Pixels in Each Cluster and Find Average Color
# -----------------------
# This part is added based on your request. It provides a quantitative
# summary of the clustering results.
unique_labels, counts = np.unique(labels, return_counts=True)
print("\n--- Cluster Pixel Counts ---")
for label, count in zip(unique_labels, counts):
    print(f"Cluster {label}: {count} pixels")

print("\n--- Average Cluster Colors (Centroids) ---")
# The centroids are already computed by MiniBatchKMeans, so we just
# use that array for efficiency.
for i in range(n_top_level_clusters):
    print(f"Cluster {i} Average Color (R,G,B): {centroids[i].astype(int)}")


# -----------------------
# Step 3: Visualize the Clustered (Segmented) Image
# -----------------------
# This visualizes the result of the top-level clustering, where each pixel
# is replaced with its cluster's average color.
segmented_img = centroids[labels].reshape(200, 200, 3)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img_array)
plt.title("Original Image")
plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(segmented_img.astype(np.uint8))
plt.title(f"Clustered Image ({n_top_level_clusters} Top-level Clusters)")
plt.axis("off")
plt.show()

# -----------------------
# Step 4: Create the High-Level Dendrogram of Centroids
# -----------------------
# This dendrogram provides a summary of the hierarchical relationships between
# the main color groups. It is computationally feasible because it only
# operates on a small number of centroids (9 in this case).
print("\n--- Generating Dendrogram of Cluster Centroids ---")
start_time = time.time()
linked_centroids = linkage(centroids, method='ward')
print(f"Linkage on centroids took {time.time() - start_time:.2f} seconds.")

plt.figure(figsize=(12, 6))
dendrogram(linked_centroids,
           orientation='top',
           distance_sort='descending',
           labels=[f'Cluster {i}' for i in range(n_top_level_clusters)],
           leaf_rotation=45,
           show_leaf_counts=True)
plt.title("Dendrogram of Cluster Centroids")
plt.xlabel("Cluster Label")
plt.ylabel("Distance")
plt.show()

# -----------------------
# Step 5: Generate Detailed Sub-Dendrograms for Each Top-Level Cluster
# -----------------------
# This is a powerful analysis. We loop through each of the 9 main clusters
# and perform a detailed hierarchical analysis on a manageable subset of its
# pixels, providing fine-grained insight into each cluster's internal structure.
max_dendrogram_pixels = 500 # Sets a limit for dendrogram generation speed
print("\n--- Generating Sub-Dendrograms for Each Top-Level Cluster ---")

for cluster_id in range(n_top_level_clusters):
    print(f"\nProcessing sub-hierarchy for Cluster {cluster_id}...")

    sub_cluster_pixels = pixels[labels == cluster_id]
    num_pixels = len(sub_cluster_pixels)
    
    if num_pixels == 0:
        print(f"  Cluster {cluster_id} is empty. Skipping.")
        continue
    
    # Take a random subset of pixels if the cluster is too large
    if num_pixels > max_dendrogram_pixels:
        subset_idx = np.random.choice(num_pixels, max_dendrogram_pixels, replace=False)
        dendrogram_pixels = sub_cluster_pixels[subset_idx]
        print(f"  Using a subset of {max_dendrogram_pixels} pixels for the dendrogram.")
    else:
        dendrogram_pixels = sub_cluster_pixels
        print(f"  Using all {num_pixels} pixels for the dendrogram.")

    # Perform the hierarchical clustering on this focused subset
    start_time = time.time()
    sub_linked = linkage(dendrogram_pixels, method='ward')
    print(f"  Hierarchical clustering on sub-cluster took {time.time() - start_time:.2f} seconds.")

    # Visualize the sub-dendrogram
    plt.figure(figsize=(12, 6))
    dendrogram(sub_linked,
               orientation='top',
               distance_sort='descending',
               show_leaf_counts=True)
    plt.title(f"Detailed Dendrogram for a subset of Pixels from Cluster {cluster_id}")
    plt.xlabel("Pixel Index")
    plt.ylabel("Distance")
    plt.show()

print("\n--- All visualizations generated successfully. ---")