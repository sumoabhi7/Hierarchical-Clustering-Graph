import numpy as np
import networkx as nx
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt

# 1. Create graph
G = nx.Graph()
edges = [('A', 'B'), ('A', 'D'), ('B', 'C'), ('B', 'E'), ('D', 'E')]
G.add_edges_from(edges)

# 2. Compute Jaccard distance
nodes = sorted  (G.nodes())
n = len(nodes)
distance_matrix = np.zeros((n, n))
for i in range(n):
    for j in range(i+1,n):
        if i != j:
            ni = set(G.neighbors(nodes[i]))
            nj = set(G.neighbors(nodes[j]))
            intersection = len(ni & nj)
            union = len(ni | nj)
            distance = 1 - (intersection / union) if union != 0 else 1
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance

# 3. Hierarchical clustering
condensed_dist = squareform(distance_matrix)
Z = linkage(condensed_dist, method='average')

plt.figure(figsize=(8,4))
dendrogram(Z, labels=nodes, orientation='top')
plt.title("Hierarchical Clustering Dendrogram (Jaccard Distance)")
plt.xlabel("Nodes")
plt.ylabel("Distance")
plt.show()

# 4. Extract clusters
k = 2   # Desired number of clusters
clusters = fcluster(Z, t=k, criterion='maxclust')
node_cluster = dict(zip(nodes, clusters))

# 5. Visualize
plt.figure(figsize=(6,4))
colors = ['red' if node_cluster[node] == 1 else 'blue' for node in nodes]
nx.draw(G, with_labels=True, node_color=colors , node_size = 800 , font_weight = 'bold')
plt.title("Graph Clustering Results")
plt.show()