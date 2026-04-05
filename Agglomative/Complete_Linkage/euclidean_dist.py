import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.pyplot as plt

# Sample data
X = np.array([[1, 2], [1.5, 2], [2, 1], 
              [5, 5], [5.5, 5], [6, 6]])

# Complete linkage clustering (correct usage)
Z = linkage(X, method='complete', metric='euclidean')  # Note 'complete' in quotes

# Plot dendrogram
plt.figure(figsize=(10, 5))
dendrogram(Z,
           labels=['P1', 'P2', 'P3', 'P4', 'P5', 'P6'],
           orientation='top')
plt.title('Complete Linkage Clustering')
plt.ylabel('Distance')
plt.show()

# Get cluster assignments
clusters = fcluster(Z, t=2, criterion='maxclust')
print("Cluster assignments:", clusters)

# 6. Visualize Original Data with Clusters
plt.figure(figsize=(6, 6))
plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis', s=100, edgecolor='k')
plt.title('Data Points with Cluster Assignments')
plt.xlabel('X-coordinate')
plt.ylabel('Y-coordinate')
for i, (x, y) in enumerate(X):
    plt.text(x+0.1, y+0.1, f'P{i+1}', fontsize=10)  # Label points
plt.colorbar(label='Cluster ID')
plt.grid(True)
plt.show()