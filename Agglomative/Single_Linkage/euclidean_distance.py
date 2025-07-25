import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler # For data scaling

# Sample Data
X = np.array([[1, 2],
              [1.5, 1.8],
              [5, 8],
              [8, 8],
              [1, 0.6],
              [9, 11]])

print("Original Data (X):\n", X)

# --- Data Preprocessing (Scaling is often crucial!) ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("\nScaled Data (X_scaled):\n", X_scaled)

# --- Perform Agglomerative Clustering using scipy ---
# 'method': linkage criterion (e.g., 'ward', 'single', 'complete', 'average')
# 'metric': distance metric (e.g., 'euclidean', 'cosine')
# Ward's method is generally good for compact, spherical clusters and works well with Euclidean distance.
Z = linkage(X_scaled, method='ward', metric='euclidean')

print("\nLinkage Matrix (Z):\n", Z)
print("Each row in Z represents a merge:")
print("[cluster_id_1, cluster_id_2, distance_of_merge, num_original_points_in_new_cluster]")

# --- Plot the Dendrogram ---
plt.figure(figsize=(12, 8)) # Adjust figure size for better readability
dendrogram(
    Z,
    labels=np.arange(len(X)), # Use original indices as labels
    leaf_rotation=45,         # Rotate leaf labels for better readability
    leaf_font_size=10,        # Adjust font size for leaf labels
    show_contracted=True      # Show contracted nodes (for larger trees)
)
plt.title('Hierarchical Clustering Dendrogram (Ward Linkage, Euclidean Distance)')
plt.xlabel('Data Point Index or (Number of points in cluster)')
plt.ylabel('Distance (Euclidean)')
plt.grid(True, linestyle='--', alpha=0.6) # Add a grid for better readability
plt.tight_layout() # Adjust layout to prevent labels from overlapping
plt.show()

# --- Optional: Extract Clusters (Cutting the Dendrogram) ---
# Option 1: Specify a distance threshold (e.g., cut where distance is less than 3)
max_d = 3
clusters_by_distance = fcluster(Z, max_d, criterion='distance')
print(f"\nClusters extracted by distance threshold ({max_d}): {clusters_by_distance}")

# Option 2: Specify a desired number of clusters (e.g., 2 clusters)
num_clusters = 2
clusters_by_num = fcluster(Z, num_clusters, criterion='maxclust')
print(f"Clusters extracted by desired number of clusters ({num_clusters}): {clusters_by_num}")