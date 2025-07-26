import math
import itertools
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram

# --- 1. Helper Functions (Basic Math) ---
def euclidean_distance(point1, point2):
    """Calculates the Euclidean distance between two points."""
    distance = 0
    for i in range(len(point1)):
        distance += (point1[i] - point2[i]) ** 2
    return math.sqrt(distance)

def calculate_centroid(cluster):
    """Calculates the centroid (average point) of a cluster."""
    if not cluster: return None
    num_points = len(cluster)
    num_dimensions = len(cluster[0])
    centroid = [0] * num_dimensions
    for point in cluster:
        for i in range(num_dimensions):
            centroid[i] += point[i]
    return tuple(c / num_points for c in centroid)

# --- 2. Function for Splitting Criterion ---
def splitting_criterion(cluster1, cluster2):
    """Measures how good a split is by the distance between centroids."""
    if not cluster1 or not cluster2: return 0
    return euclidean_distance(calculate_centroid(cluster1), calculate_centroid(cluster2))

# --- 3. Function to Perform the Split ---
def find_best_split(cluster):
    """Finds the best partition of a cluster into two non-empty sub-clusters."""
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

# --- 4. Main Recursive Splitting Function ---
def perform_recursive_split(cluster, max_depth, current_depth=0):
    """Performs divisive clustering recursively."""
    if len(cluster) <= 2 or current_depth >= max_depth:
        return {'points': cluster, 'children': [], 'id': f'L{id(cluster)}'}

    child1_points, child2_points = find_best_split(cluster)

    if not child1_points or not child2_points:
        return {'points': cluster, 'children': [], 'id': f'L{id(cluster)}'}
    
    node_id = f'N{current_depth}-{id(cluster)}'
    
    child1_node = perform_recursive_split(child1_points, max_depth, current_depth + 1)
    child2_node = perform_recursive_split(child2_points, max_depth, current_depth + 1)
    
    return {'points': cluster, 'children': [child1_node, child2_node], 'id': node_id}

# --- 5. Helper and Visualization Functions ---

def get_leaf_clusters(node):
    """Traverses the hierarchy tree and returns a flat list of final leaf clusters."""
    if not node['children']:
        return [node['points']]
    
    leaves = []
    for child in node['children']:
        leaves.extend(get_leaf_clusters(child))
    return leaves

def visualize_clusters(final_clusters):
    """Generates a scatter plot showing the final clusters."""
    plt.figure(figsize=(10, 7))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'] # Default matplotlib colors
    
    for i, cluster in enumerate(final_clusters):
        if cluster: 
            x_coords = [point[0] for point in cluster]
            y_coords = [point[1] for point in cluster]
            plt.scatter(x_coords, y_coords, color=colors[i % len(colors)], s=100, label=f'Cluster {i+1}')

    plt.title('Divisive Clustering Results (Scatter Plot)', fontsize=16)
    plt.xlabel('X-axis', fontsize=12)
    plt.ylabel('Y-axis', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

def show_dendrogram_plot(hierarchy_node, original_data):
    """Builds a linkage matrix with meaningful distances and displays a scipy dendrogram."""
    linkage_list = []
    node_count = len(original_data)
    
    def build_linkage(node):
        nonlocal node_count
        if not node['children']:
            if len(node['points']) == 1:
                return original_data.index(node['points'][0]), 1
            else: # If a leaf has 2 points, create a merge for them
                p1_idx = original_data.index(node['points'][0])
                p2_idx = original_data.index(node['points'][1])
                dist = euclidean_distance(node['points'][0], node['points'][1])
                linkage_list.append([p1_idx, p2_idx, dist, 2])
                internal_node_index = node_count
                node_count += 1
                return internal_node_index, 2

        child1_node = node['children'][0]
        child2_node = node['children'][1]
        
        child1_idx, count1 = build_linkage(child1_node)
        child2_idx, count2 = build_linkage(child2_node)
        
        distance = splitting_criterion(child1_node['points'], child2_node['points'])
        linkage_list.append([child1_idx, child2_idx, distance, count1 + count2])
        
        internal_node_index = node_count
        node_count += 1
        return internal_node_index, count1 + count2
        
    build_linkage(hierarchy_node)
    
    if not linkage_list:
        print("Cannot generate dendrogram, no splits were made.")
        return

    plt.figure(figsize=(10, 7))
    dendrogram(
        np.array(linkage_list, dtype=float),
        labels=[str(i) for i in range(len(original_data))],
        orientation='top'
    )
    
    plt.title("Divisive Clustering Dendrogram")
    plt.xlabel("Data Point")
    plt.ylabel("Distance")
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.show()

# --- Main Execution ---
if __name__ == "__main__":
    # Data structured to match the target dendrogram
    data = [
        (10, 10),  # Point 0
        (11, 10),  # Point 1 
        (20, 20),  # Point 2
        (50, 50),  # Point 3
        (40, 40),  # Point 4
        (50, 51)   # Point 5
    ]
    
    MAX_SPLIT_DEPTH = 3 
    
    print("Starting Divisive Clustering...")
    hierarchy = perform_recursive_split(data, max_depth=MAX_SPLIT_DEPTH)
    
    # 1. Visualize the final clusters on a scatter plot
    print("Displaying scatter plot of final clusters...")
    final_clusters = get_leaf_clusters(hierarchy)
    visualize_clusters(final_clusters)
    
    # 2. Visualize the hierarchy as a dendrogram
    print("Displaying dendrogram...")
    show_dendrogram_plot(hierarchy, data)