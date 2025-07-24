import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, cophenet
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt

# 1. Input Data
X = np.array([[1, 1], [2, 2], [3, 3], 
              [10, 1], [11, 2], [12, 3]])

# 2. Single Linkage with MANHATTAN distance
Z = linkage(X, method='single', metric='cityblock')  # cityblock = Manhattan

# 3. Calculate optimal color threshold (25th percentile of distances)
threshold = np.percentile(Z[:, 2], 25)

# 4. Plot Enhanced Dendrogram
plt.figure(figsize=(10, 5))
dendrogram(Z, 
           labels=['P1', 'P2', 'P3', 'P4', 'P5', 'P6'],
           orientation='top',
           color_threshold=threshold,  # Dynamic threshold
           above_threshold_color='grey')
plt.title('Single Linkage Dendrogram (Manhattan Distance)')  # Corrected title
plt.ylabel('Manhattan Distance')  # Corrected axis label
plt.xlabel('Data Points')
plt.grid(axis='y', linestyle=':')
plt.show()

# 5. Extract Clusters
k = 2
clusters = fcluster(Z, t=k, criterion='maxclust')
print(f"Cluster Assignments: {clusters}")

# 6. Visualize Clusters
plt.figure(figsize=(6, 6))
plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis', s=100, edgecolor='k')
plt.title('Clusters (Manhattan Distance)')
plt.xlabel('X-coordinate')
plt.ylabel('Y-coordinate')

# Annotate points
for i, (x, y) in enumerate(X):
    plt.text(x+0.1, y+0.1, f'P{i+1}', fontsize=10)

plt.colorbar(label='Cluster ID')
plt.grid(True)
plt.show()