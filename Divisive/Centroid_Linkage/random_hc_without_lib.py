import networkx as nx
import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
import random

# --- Helper Functions ---

def get_cluster_diameter(cluster, paths):
    """Calculates the longest shortest-path between any two nodes in a cluster."""
    if len(cluster) < 2:
        return 0
    max_dist = 0
    for node1, node2 in combinations(cluster, 2):
        distance = paths[node1].get(node2, -1)
        if distance > max_dist:
            max_dist = distance
    return max_dist

def avg_dist(node, group, paths):
    """Calculates the average distance from a node to all other nodes in a group."""
    other_nodes = [n for n in group if n != node]
    if not other_nodes:
        return 0
    total_dist = 0
    for target_node in other_nodes:
        # Use a large number for disconnected nodes
        total_dist += paths[node].get(target_node, 999)
    return total_dist / len(other_nodes)

def split_cluster(cluster_to_split, paths):
    """Splits a single cluster into two using the DIANA method."""
    if len(cluster_to_split) < 2:
        return [cluster_to_split]

    ci = cluster_to_split.copy()
    cj = []

    # 1. Find the node with the max average distance to start the splinter group
    max_avg_dist = -1
    splinter_node = -1
    for node in ci:
        current_dist = avg_dist(node, ci, paths)
        if current_dist > max_avg_dist:
            max_avg_dist = current_dist
            splinter_node = node
            
    ci.remove(splinter_node)
    cj.append(splinter_node)

    # 3. Iteratively move nodes that are closer to cj than to ci
    while True:
        attractions = {}
        if not ci: break
        for node in ci:
            dist_to_ci = avg_dist(node, ci, paths)
            dist_to_cj = avg_dist(node, cj, paths)
            attractions[node] = dist_to_ci - dist_to_cj
        
        node_to_move = max(attractions, key=attractions.get)
        max_attraction = attractions[node_to_move]

        if max_attraction > 0:
            ci.remove(node_to_move)
            cj.append(node_to_move)
        else:
            break
    return [ci, cj]


# --- Main Algorithm & History Logging ---

# 1. Setup
N = 10
P = 0.3
adj_matrix = np.zeros((N, N))

# Fill the matrix with random edges
for i in range(N):
    for j in range(i + 1, N):
        if random.random() < P: # Use random.random() as it respects the seed
            adj_matrix[i, j] = 1
            adj_matrix[j, i] = 1 # Symmetric for an undirected graph

# Create the graph from the completed matrix
G = nx.from_numpy_array(adj_matrix)

# Optional: Show the initial raw graph
plt.figure(figsize=(8, 6))
nx.draw(G, with_labels=True, node_color='skyblue')
plt.title(f"Initial Random Graph (N={N})")
plt.show()

all_pairs_paths = dict(nx.shortest_path_length(G))
nodes = sorted(G.nodes())

# 2. Start with a single cluster and an empty history log
clusters = [list(G.nodes())]
split_history = []
print("--- Initial State ---")
print(f"Starting with one large cluster of {len(clusters[0])} nodes:")
print(clusters[0])
print("-" * 40 + "\n")


iteration_count = 1
# 3. Main loop: continue as long as there's a cluster to split
while True:
    diameters = [get_cluster_diameter(c, all_pairs_paths) for c in clusters]
    max_diameter = max(diameters)

    if max_diameter == 0:
        print("--- All clusters have a diameter of 0. Process complete. ---")
        break
        
    index_to_split = diameters.index(max(diameters))
    cluster_to_split = clusters.pop(index_to_split)
    
    print(f"--- Iteration {iteration_count} ---")
    print(f"Choosing cluster to split (Diameter: {max_diameter})...")

    new_clusters = split_cluster(cluster_to_split, all_pairs_paths)
    clusters.extend(new_clusters)
    
    print(f"Split into: {new_clusters[0]} and {new_clusters[1]}")
    
    # Log the split for the dendrogram
    split_history.append((
        tuple(sorted(cluster_to_split)),
        tuple(sorted(new_clusters[0])),
        tuple(sorted(new_clusters[1])),
        max_diameter
    ))
    
    # Show all clusters at the end of the iteration
    print("\nAll clusters at end of this iteration:")
    for cluster in sorted(clusters):
        print(f"  - {cluster}")
    print("-" * 40)
    
    iteration_count += 1


# --- Dendrogram Generation from Split History ---

print("\nGenerating dendrogram from split history...")
cluster_to_id = {tuple([n]): i for i, n in enumerate(nodes)}
n_points = len(nodes)
next_id = n_points
linkage_matrix = []

# Process splits in reverse to build the linkage matrix
for parent, child1, child2, distance in reversed(split_history):
    id1 = cluster_to_id[child1]
    id2 = cluster_to_id[child2]
    
    linkage_matrix.append([id1, id2, distance, len(parent)])
    cluster_to_id[tuple(sorted(parent))] = next_id
    next_id += 1

linkage_matrix = np.array(linkage_matrix, dtype=np.double)

# --- Final Dendrogram Plotting ---
plt.figure(figsize=(12, 8))
dendrogram(
    linkage_matrix,
    labels=nodes,
    orientation='top',
    distance_sort='descending'
)
plt.title(f'Dendrogram from Custom Divisive Clustering (N={N})')
plt.xlabel('Node')
plt.ylabel('Distance (Diameter at Split)')
plt.show()