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
    connected_pairs = 0
    for node1, node2 in combinations(cluster, 2):
        distance = paths[node1].get(node2, float('inf'))
        if distance != float('inf'):
            max_dist = max(max_dist, distance)
            connected_pairs += 1
    return max_dist if connected_pairs > 0 else float('inf')

def avg_dist(node, group, paths):
    """Calculates the average distance from a node to all other nodes in a group."""
    other_nodes = [n for n in group if n != node]
    if not other_nodes:
        return 0
    total_dist = 0
    valid_pairs = 0
    for target_node in other_nodes:
        distance = paths[node].get(target_node, float('inf'))
        if distance != float('inf'):
            total_dist += distance
            valid_pairs += 1
    return total_dist / valid_pairs if valid_pairs > 0 else float('inf')

def split_cluster(cluster_to_split, paths):
    """
    Splits a single cluster into two using the DIANA method.
    This version does NOT use spectral clustering.
    """
    if len(cluster_to_split) < 2:
        return [cluster_to_split]

    # 1. Find the node with the highest average distance to others in the cluster.
    # This node will start the new "splinter" group.
    max_avg_dist = -1
    splinter_node_starter = -1
    for node in cluster_to_split:
        current_avg_dist = avg_dist(node, cluster_to_split, paths)
        if current_avg_dist > max_avg_dist:
            max_avg_dist = current_avg_dist
            splinter_node_starter = node
    
    # The original cluster remains, and a new one is created.
    old_party = cluster_to_split.copy()
    splinter_group = []

    # 2. Move the starter node to the splinter group.
    old_party.remove(splinter_node_starter)
    splinter_group.append(splinter_node_starter)

    # 3. Iteratively move nodes from the old party to the splinter group
    #    if they are closer to the splinter group than their own.
    while True:
        node_to_move = None
        max_attraction = -float('inf')

        if not old_party: # Stop if the old party is empty
            break
            
        for node in old_party:
            dist_to_old = avg_dist(node, old_party, paths)
            dist_to_splinter = avg_dist(node, splinter_group, paths)
            
            # If the difference is positive, the node is "attracted" to the splinter group.
            attraction = dist_to_old - dist_to_splinter
            if attraction > max_attraction:
                max_attraction = attraction
                node_to_move = node
        
        # If the most attracted node has a positive attraction, move it.
        # Otherwise, no more nodes should be moved.
        if max_attraction > 0:
            old_party.remove(node_to_move)
            splinter_group.append(node_to_move)
        else:
            break # Stop if no more nodes are attracted to the splinter group

    return [old_party, splinter_group]

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
#random.seed(42)    # By commenting this out, the graph and as well as dendogram will be truly random on each run

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

plot_graph(G, title=f"Initial Graph for Clustering (Nodes: {G.number_of_nodes()})")

all_pairs_paths = dict(nx.shortest_path_length(G, weight='weight'))
nodes = sorted(G.nodes())

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
    max_diameter = max(diameters)

    if max_diameter == 0:
        print("--- All remaining clusters have 0 diameter. Process complete. ---")
        break
        
    index_to_split = diameters.index(max_diameter)
    cluster_to_split = clusters.pop(index_to_split)
    
    print(f"--- Iteration {iteration_count} ---")
    print(f"Choosing cluster of size {len(cluster_to_split)} to split (Diameter: {max_diameter:.2f})...")

    new_clusters = split_cluster(cluster_to_split, all_pairs_paths)
    
    # Add new clusters if they are not empty
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

# --- Dendrogram Generation ---
if not split_history:
    print("\nNo splits were made. Cannot generate dendrogram.")
else:
    print("\nGenerating dendrogram from split history...")
    cluster_to_id = {}
    next_id = 0
    
    # Correctly identify leaf nodes from the final cluster list
    final_leaf_clusters = [tuple(sorted(c)) for c in clusters]
    for key in final_leaf_clusters:
        if key not in cluster_to_id:
            cluster_to_id[key] = next_id
            next_id += 1
            
    linkage_matrix = []
    for parent, child1, child2, distance in reversed(split_history):
        # Children might be multi-node leaves
        id1 = cluster_to_id[child1]
        id2 = cluster_to_id[child2]
        
        # The number of original observations in the new cluster
        num_obs = len(parent)
        
        linkage_matrix.append([id1, id2, distance, num_obs])
        cluster_to_id[parent] = next_id
        next_id += 1

    linkage_matrix = np.array(linkage_matrix, dtype=np.double)
    
    # We must ensure the linkage matrix is valid for scipy
    if linkage_matrix.shape[0] + 1 != len(final_leaf_clusters):
        print("Cannot plot dendrogram: The number of splits does not match the number of final clusters.")
    else:
        plt.figure(figsize=(15, 10))
        dendrogram(linkage_matrix)
        plt.title(f'Hierarchical Clustering Dendrogram')
        plt.xlabel('Cluster Index')
        plt.ylabel('Cluster Diameter at Split')
        plt.show()