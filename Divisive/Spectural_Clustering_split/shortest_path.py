import networkx as nx
import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
import random
# Import added for the new splitting method
from sklearn.cluster import SpectralClustering

# --- Helper Functions ---

def get_cluster_diameter(cluster, paths):
    """Calculates the longest shortest-path between any two nodes in a cluster."""
    if len(cluster) < 2:
        return 0
    max_dist = 0
    connected_pairs = 0
    for node1, node2 in combinations(cluster, 2):
        distance = paths[node1].get(node2, float('inf'))
        if distance != float('inf'):
            max_dist = max(max_dist, distance)
            connected_pairs += 1
    return max_dist if connected_pairs > 0 else float('inf')

# The avg_dist function is no longer needed for this method

def split_cluster(cluster_to_split, paths):
    """
    Splits a single cluster into two using Spectral Clustering, which is
    a graph-based equivalent of bisecting k-means.
    """
    if len(cluster_to_split) < 2:
        return [cluster_to_split]

    size = len(cluster_to_split)
    
    # Create a distance sub-matrix for only the nodes in the current cluster
    distances = np.full((size, size), float('inf'))
    for i, n1 in enumerate(cluster_to_split):
        for j, n2 in enumerate(cluster_to_split):
            if i == j:
                distances[i, j] = 0
            else:
                distances[i, j] = paths[n1].get(n2, float('inf'))

    # If the cluster is internally disconnected, we can't proceed
    if np.all(np.isinf(distances[np.triu_indices(size, k=1)])):
        return [cluster_to_split] 
        
    # Handle any potential disconnected pairs before calculating similarity
    max_finite_dist = np.max(distances[np.isfinite(distances)])
    distances[np.isinf(distances)] = max_finite_dist * 1.5 

    # Convert the distance matrix to a similarity (or affinity) matrix
    # The gamma parameter can be tuned; 1.0 is a reasonable default
    similarities = np.exp(-distances**2 / (2. * 1.0**2))

    try:
        # Use SpectralClustering to find 2 clusters
        sc = SpectralClustering(2, affinity='precomputed', random_state=42)
        labels = sc.fit_predict(similarities)
        
        cluster1 = [node for node, label in zip(cluster_to_split, labels) if label == 0]
        cluster2 = [node for node, label in zip(cluster_to_split, labels) if label == 1]
        
        # Ensure the split was successful and didn't create an empty cluster
        if not cluster1 or not cluster2:
            return [cluster_to_split]

        return [cluster1, cluster2]
    except Exception as e:
        print(f"  > Spectral clustering failed: {e}")
        return [cluster_to_split]


def plot_graph(G , title="Graph Visualization"):
    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(G,seed=42)     # Use a spring layout with a fixed seed for reproducible plots
    weights = [G[u][v].get('weight',1.0) for u , v in G.edges()]     # Check if edges have weights to visualize them by thickness
    nx.draw(G , pos ,
            with_labels = True,
            node_color = 'skyblue',
            node_size =  700,
            edge_color = 'gray',
            width = [w * 1.5 for w in weights]    # Make line width proportional to weight
            )   
    plt.title(title, fontsize=16)
    plt.show()


# --- Main Algorithm ---

# 1. Setup
N = 30
P = 0.4
random.seed(42)

# Create weighted random graph
adj_matrix = np.zeros((N, N))
for i in range(N):
    for j in range(i + 1, N):
        if random.random() < P:
            adj_matrix[i, j] = random.uniform(0.5, 2.0)
            adj_matrix[j, i] = adj_matrix[i, j]

G = nx.from_numpy_array(adj_matrix)
if not nx.is_connected(G):
    largest_cc = max(nx.connected_components(G), key=len)
    G = G.subgraph(largest_cc).copy()

all_pairs_paths = dict(nx.shortest_path_length(G, weight='weight'))
nodes = sorted(G.nodes())

plot_graph(G, title=f"Initial Graph for Clustering (Nodes: {G.number_of_nodes()})")

# 2. Start with one cluster containing all nodes
clusters = [list(G.nodes())]
split_history = []
print(f"--- Starting with one cluster of {len(clusters[0])} nodes ---")

# 3. Divisive clustering
iteration_count = 1
while True:
    if len(clusters) >= N: # Stop if every node is its own cluster
        break
        
    diameters = [get_cluster_diameter(c, all_pairs_paths) for c in clusters]
    
    # Find a splittable cluster (diameter > 0)
    splittable_clusters_indices = [i for i, d in enumerate(diameters) if d > 0]
    if not splittable_clusters_indices:
        print("--- All remaining clusters have 0 diameter. Process complete. ---")
        break
        
    # Choose the one with the largest diameter to split
    index_to_split = max(splittable_clusters_indices, key=lambda i: diameters[i])
    cluster_to_split = clusters.pop(index_to_split)
    max_diameter = diameters[index_to_split]
    
    print(f"--- Iteration {iteration_count} ---")
    print(f"Choosing cluster of size {len(cluster_to_split)} to split (Diameter: {max_diameter:.2f})...")

    new_clusters = split_cluster(cluster_to_split, all_pairs_paths)
    
    for nc in new_clusters:
        if nc: clusters.append(nc)
    
    print(f"Split into clusters of size: {[len(c) for c in new_clusters]}")
    
    if len(new_clusters) == 2 and new_clusters[0] and new_clusters[1]:
      split_history.append((
          tuple(sorted(cluster_to_split)),
          tuple(sorted(new_clusters[0])),
          tuple(sorted(new_clusters[1])),
          max_diameter
      ))
    
    iteration_count += 1

# --- Dendrogram Generation (Same as before) ---
if not split_history:
    print("\nNo splits were made. Cannot generate dendrogram.")
else:
    print("\nGenerating dendrogram from split history...")
    cluster_to_id = {}
    cluster_leaf_count = {}
    next_id = 0
    
    final_leaf_clusters = [tuple(sorted(c)) for c in clusters]
    for key in final_leaf_clusters:
        if key not in cluster_to_id:
            cluster_to_id[key] = next_id
            cluster_leaf_count[key] = 1
            next_id += 1
            
    linkage_matrix = []
    for parent, child1, child2, distance in reversed(split_history):
        id1 = cluster_to_id[child1]
        id2 = cluster_to_id[child2]
        
        count1 = cluster_leaf_count[child1]
        count2 = cluster_leaf_count[child2]
        new_leaf_count = count1 + count2
        
        cluster_leaf_count[parent] = new_leaf_count
        
        linkage_matrix.append([id1, id2, distance, new_leaf_count])
        cluster_to_id[parent] = next_id
        next_id += 1

    linkage_matrix = np.array(linkage_matrix, dtype=np.double)

    if linkage_matrix.shape[0] + 1 != len(final_leaf_clusters):
         print("Cannot plot dendrogram: The number of splits does not match the number of final clusters.")
    else:
        plt.figure(figsize=(15, 10))
        dendrogram(
            linkage_matrix,
            orientation='top',
            distance_sort='descending',
            show_leaf_counts=True
        )
        plt.title(f'Hierarchical Clustering Dendrogram (Bisecting Method)')
        plt.xlabel('Cluster Index')
        plt.ylabel('Cluster Diameter at Split')
        plt.show()