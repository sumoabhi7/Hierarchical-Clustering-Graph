from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# -----------------------
# Step 1: Load and preprocess image
# -----------------------
image_path = r"dummy_img.png"  # Change to your image path
img = Image.open(image_path).convert('RGB').resize((200, 200))
img_array = np.array(img)  # Shape: (200, 200, 3)
pixels = img_array.reshape(-1, 3)  # Shape: (40000, 3)

print(f"Image converted to RGB. Shape: {img_array.shape}, Pixels: {pixels.shape}")

# -----------------------
# Step 2: Create dendrogram (subset for speed)
# -----------------------
subset_size = 500
subset_idx = np.random.choice(len(pixels), subset_size, replace=False)
subset_pixels = pixels[subset_idx]

linked = linkage(subset_pixels, method='ward')  # Uses Euclidean Distance method

plt.figure(figsize=(10, 5))
dendrogram(linked,
           orientation='top',
           distance_sort='descending',
           show_leaf_counts=True)
plt.title("Dendrogram (Subset of Pixels)")
plt.show()

# -----------------------
# Step 3: Apply hierarchical clustering
# -----------------------
n_clusters = 5  # Change as needed
clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
labels = clustering.fit_predict(pixels)

print("Clustering complete!")

# Step 3.1 : Counting Pixels in Each Cluster

unique_labels , counts = np.unique(labels , return_counts=True)

print("\n--- Cluster Pixel Counts ---")
for label , count in zip(unique_labels , counts):
    print(f"Cluster {label} : {count} pixels")

# Step 3.2 : Finding the Average Color of Each Cluster

print("\n--- Average Cluster Colors (Centroids) ---")
for i in range(n_clusters):
    # Find all pixels belonging to the current cluster
    cluster_pixels = pixels[labels == i]
    
    # Calculate the average color (R, G, B)
    average_color = cluster_pixels.mean(axis=0)
    
    print(f"Cluster {i} Average Color (R,G,B): {average_color.astype(int)}")

# -----------------------
#Step 4: Create segmented image (IMPROVED)
# -----------------------
# Create a palette from the calculated average colors (centroids)
centroid_colors = np.zeros((n_clusters, 3))
for i in range(n_clusters):
    centroid_colors[i] = pixels[labels == i].mean(axis=0)

# Use the centroid palette to create the segmented image
segmented_img = centroid_colors[labels].reshape(200, 200, 3)

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(img_array)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(segmented_img.astype(np.uint8))
plt.title(f"Clustered Image ({n_clusters} Clusters)")
plt.axis("off")

plt.show()