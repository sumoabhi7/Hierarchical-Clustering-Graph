import math
import itertools
import random
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from scipy.cluster.hierarchy import dendrogram

# --- 1. Graph Generation and Visualization ---
def generate_random_graph_matrix(num_nodes):
    matrix = [[0] * num_nodes for _ in range(num_nodes)]
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if random.random() > 0.5:
                matrix[i][j] = 1
                matrix[j][i] = 1
    return matrix

def visualize_graph(adjacency_matrix):
    G = nx.from_numpy_array(np.array(adjacency_matrix))
    plt.figure(figsize=(8, 8))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_color='#87CEEB', node_size=700, font_size=10, font_weight='bold')
    plt.title("Generated Random Graph")
    plt.show()

# --- 2. Core Clustering Logic (From Scratch) ---
def euclidean_distance(point1, point2):
    distance = 0
    for i in range(len(point1)):
        distance += (point1[i] - point2[i]) ** 2
    return math.sqrt(distance)

def calculate_centroid(cluster):
    if not cluster: return None
    num_points = len(cluster)
    num_dimensions = len(cluster[0])
    centroid = [0] * num_dimensions
    for point in cluster:
        for i in range(num_dimensions):
            centroid[i] += point[i]
    return tuple(c / num_points for c in centroid)

def splitting_criterion(cluster1, cluster2):
    if not cluster1 or not cluster2: return 0
    return euclidean_distance(calculate_centroid(cluster1), calculate_centroid(cluster2))

def find_best_split(cluster):
    best_split = ([], [])
    max_score = -1
    n = len(cluster)
    if n < 2: return cluster, []
    for i in range(1, n // 2 + 1):
        for subset1_indices in itertools.combinations(range(n), i):
            sub_cluster1 = [cluster[j] for j in subset1_indices]
            sub_cluster2 = [cluster[j] for j in range(n) if j not in subset1_indices]
            score = splitting_criterion(sub_cluster1, sub_cluster2)
            if score > max_score:
                max_score = score
                best_split = (sub_cluster1, sub_cluster2)
    return best_split

def perform_recursive_split(cluster, max_depth, current_depth=0):
    if len(cluster) <= 1 or current_depth >= max_depth:
        return {'points': cluster, 'children': []}
    child1_points, child2_points = find_best_split(cluster)
    if not child1_points or not child2_points:
        return {'points': cluster, 'children': []}
    child1_node = perform_recursive_split(child1_points, max_depth, current_depth + 1)
    child2_node = perform_recursive_split(child2_points, max_depth, current_depth + 1)
    return {'points': cluster, 'children': [child1_node, child2_node]}

# --- 3. Visualization and Helper Functions ---

def get_leaf_clusters(node):
    """Traverses the hierarchy tree and returns a flat list of final leaf clusters."""
    if not node['children']:
        return [node['points']]
    leaves = []
    for child in node['children']:
        leaves.extend(get_leaf_clusters(child))
    return leaves

def show_dendrogram_plot(hierarchy_node, original_data):
    """Builds a compliant linkage matrix and displays a scipy dendrogram."""
    merges = []
    def collect_merges(node):
        if not node['children']:
            return
        child1_points = node['children'][0]['points']
        child2_points = node['children'][1]['points']
        distance = splitting_criterion(child1_points, child2_points)
        merges.append({
            'distance': distance,
            'cluster1': frozenset(map(tuple, child1_points)),
            'cluster2': frozenset(map(tuple, child2_points))
        })
        collect_merges(node['children'][0])
        collect_merges(node['children'][1])

    collect_merges(hierarchy_node)
    if not merges:
        print("Cannot generate dendrogram, no splits were made.")
        return

    merges.sort(key=lambda x: x['distance'])

    # --- FIX STARTS HERE ---
    
    # 1. Get the final leaf clusters from the hierarchy tree
    leaf_clusters = get_leaf_clusters(hierarchy_node)
    
    # 2. Initialize the map with all final leaf clusters, giving each a unique index.
    cluster_map = {frozenset(map(tuple, cluster)): i for i, cluster in enumerate(leaf_clusters)}
    
    next_cluster_id = len(leaf_clusters)
    linkage_list = []

    for merge in merges:
        c1_idx = cluster_map[merge['cluster1']]
        c2_idx = cluster_map[merge['cluster2']]
        new_cluster = merge['cluster1'].union(merge['cluster2'])
        linkage_list.append([c1_idx, c2_idx, merge['distance'], len(new_cluster)])
        cluster_map[new_cluster] = next_cluster_id
        next_cluster_id += 1
        
    # 3. Create labels that correspond to the leaf clusters
    point_to_idx = {point: i for i, point in enumerate(original_data)}
    leaf_labels = [str(sorted([point_to_idx[p] for p in cluster])) for cluster in leaf_clusters]

    # --- FIX ENDS HERE ---

    plt.figure(figsize=(12, 7))
    dendrogram(
        np.array(linkage_list, dtype=float),
        labels=leaf_labels,
        orientation='top'
    )
    plt.title("Divisive Clustering of Random Graph Nodes")
    plt.xlabel("Final Leaf Clusters (Node Indices)")
    plt.ylabel("Dissimilarity (Centroid Distance)")
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.show()

# --- Main Execution ---
if __name__ == "__main__":
    NUM_NODES = 8
    
    # 1. Generate the graph's adjacency matrix
    adjacency_matrix = generate_random_graph_matrix(NUM_NODES)
    
    # 2. Visualize the graph itself
    print(f"Generated a random graph with {NUM_NODES} nodes.")
    print("--> Displaying graph... PLEASE CLOSE THE PLOT WINDOW TO CONTINUE.")
    visualize_graph(adjacency_matrix)
    
    # 3. Prepare the data for clustering
    data = [tuple(row) for row in adjacency_matrix]
    
    # 4. Set clustering depth and run the algorithm
    MAX_SPLIT_DEPTH = 3 
    print("\nStarting Divisive Clustering on graph nodes...")
    hierarchy = perform_recursive_split(data, max_depth=MAX_SPLIT_DEPTH)
    
    # 5. Generate and display the dendrogram
    print("--> Generating and displaying dendrogram...")
    show_dendrogram_plot(hierarchy, data)
    
    print("\nScript finished.")