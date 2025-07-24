import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import pairwise_distances

# Step 1: Generate random adjacency matrix (0s and 1s)
n = 10  # Number of nodes
p = 0.3  # Probability of an edge
adj_matrix = np.random.choice([0, 1], size=(n, n), p=[1-p, p])
adj_matrix = np.triu(adj_matrix)  # Upper triangle
adj_matrix = adj_matrix + adj_matrix.T - np.diag(np.diag(adj_matrix))  # Symmetrize and make diagonal to zero only

# Step 2: Create a graph from the adjacency matrix
G = nx.from_numpy_array(adj_matrix)

# Step 3: Visualize the graph
plt.figure(figsize=(10, 5))
plt.subplot(121)  # Left subplot for the graph
pos = nx.circular_layout(G)  # Layout for better visualization
nx.draw(G, pos, with_labels=True, node_color='skyblue', edge_color='green', node_size=500)
plt.title("Random Graph (Edges = 1, No Edges = 0)")

# Step 4: Hierarchical clustering and dendrogram
plt.subplot(122)  # Right subplot for the dendrogram
distance_method = pairwise_distances(adj_matrix, metric="cityblock")  # Distance metric
Z = linkage(distance_method, method='average')  # Clustering method
dendrogram(Z, labels=np.arange(n), orientation='top')
plt.title("Agglomerative Clustering Dendrogram")

plt.tight_layout()
plt.show()






