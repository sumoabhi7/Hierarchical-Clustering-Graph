'''import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt


X = np.array([[1, 2], [1, 4], [1, 0],[4, 2], [4, 4], [4, 0]])

Z = linkage(X, method="single" , metric="euclidean") # Ward Distance

dendrogram(Z) #plotting the dendogram

plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Data point')
plt.ylabel('Distance')
plt.show()'''

import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, cophenet
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt

# 1. Input Data
X = np.array([[1, 1], [2, 2], [3, 3], 
              [10, 1], [11, 2], [12, 3]])

# 2. Single Linkage Clustering
Z = linkage(X, method='single', metric='euclidean')

# 3. Calculate Cophenetic Correlation (Cluster Quality)
'''coph_dist = cophenet(Z)
original_dist = pdist(X)
coph_corr = np.corrcoef(original_dist, coph_dist)[0, 1]
print(f"Cophenetic Correlation: {coph_corr:.3f}")'''

# 4. Plot Enhanced Dendrogram
plt.figure(figsize=(10, 5))
dendrogram(Z, 
           labels=['P1', 'P2', 'P3', 'P4', 'P5', 'P6'],  # Custom labels
           orientation='top',  # Horizontal plot
           color_threshold=0.5,  # Color clusters below this threshold
           above_threshold_color='grey')  # Unmerged clusters
plt.title('Single Linkage Dendrogram (Euclidean Distance)')
plt.xlabel('Distance')
plt.grid(True, linestyle=':')  # Add grid for readability
plt.show()

# 5. Extract Flat Clusters
k = 2  # Desired number of clusters
clusters = fcluster(Z, t=k, criterion='maxclust')
print(f"Cluster Assignments: {clusters}")

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