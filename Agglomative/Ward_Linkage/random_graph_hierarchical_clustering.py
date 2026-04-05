import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist, squareform

def generate_random_graph_and_cluster(num_nodes=20, edge_probability=0.3, num_clusters=None):
    """
    Generates a random graph, performs hierarchical clustering, and visualizes the results.

    Args:
        num_nodes (int): The number of nodes in the graph.
        edge_probability (float): The probability of an edge existing between any two nodes (0.0 to 1.0).
        num_clusters (int, optional): The number of clusters to form. If None, the dendrogram is shown,and you can visually determine the number.
    """

    print(f"Generating a random graph with {num_nodes} nodes and edge probability {edge_probability}...")

    # 1. Generate Adjacency Matrix (Random Edges)
    # Initialize an empty adjacency matrix
    adj_matrix = np.zeros((num_nodes, num_nodes))

    # Fill the upper triangle with random 0s or 1s based on edge_probability
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes): # Avoid self-loops and duplicate edges (undirected)
            if np.random.rand() < edge_probability:
                adj_matrix[i, j] = 1
                adj_matrix[j, i] = 1 # Make it symmetric for an undirected graph

    # 2. Make a Graph using NetworkX
    G = nx.from_numpy_array(adj_matrix)

    print(f"Graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

    # Remove isolated nodes if any (nodes with no edges), though unlikely with random generation
    # G.remove_nodes_from(list(nx.isolates(G)))
    # print(f"Graph after removing isolated nodes: {G.number_of_nodes()} nodes.")

    # 3. Prepare Data for Hierarchical Clustering
    # For graph clustering based on connectivity, we can use shortest path distances
    # or a similarity measure derived from the adjacency matrix.
    # Here, let's use a "distance" measure where 0 means connected, 1 means not connected.
    # This is effectively using the complement of the adjacency matrix as a distance.
    # We need a condensed distance matrix for scipy's linkage function.

    # Option A: Simple "distance" from adjacency (0 if connected, 1 if not)
    # This is a bit simplistic for complex clustering but works as a starting point.
    distances = 1 - adj_matrix
    # Ensure diagonal is 0 (distance to self is 0)
    np.fill_diagonal(distances, 0)
    
    # Convert the square distance matrix to a condensed (flat) form
    # This is what scipy's linkage expects for pdist.
    # Alternatively, you can use pdist on a feature matrix if your nodes had features.
    
    # Let's try to derive a "feature" for each node based on its connections
    # For example, we could use the row of the adjacency matrix as a "feature vector"
    # and then calculate Euclidean distance between these "feature vectors".
    node_features = adj_matrix # Each row represents connections of a node
    
    # Calculate pairwise Euclidean distances between node feature vectors
    dissimilarity_matrix = pdist(node_features, metric='euclidean')

    # 4. Perform Hierarchical Clustering
    # linkage_matrix contains the hierarchical clustering information
    # 'ward' method minimizes the variance of the clusters being merged.
    print("Performing hierarchical clustering...")
    Z = linkage(dissimilarity_matrix, method='ward')

    # 5. Visualize the Dendrogram
    plt.figure(figsize=(12, 7))
    plt.xlabel('Node Index')
    plt.ylabel('Distance')
    dendrogram(
        Z,
        leaf_rotation=90.,  # rotates the x axis labels
        leaf_font_size=8.,  # font size for the x axis labels
    )
    plt.grid(True)
    plt.show()

    # 6. Extract Clusters and Visualize Graph
    if num_clusters is not None:
        print(f"\nExtracting {num_clusters} clusters...")
        # fcluster extracts clusters from the linkage matrix
        # 'maxclust' criterion cuts the dendrogram to form 'num_clusters' clusters
        clusters = fcluster(Z, num_clusters, criterion='maxclust')
        
        # Map cluster IDs to node colors
        cmap = plt.cm.get_cmap('viridis', num_clusters) # Choose a colormap
        node_colors = [cmap(c_id - 1) for c_id in clusters] # -1 because cluster IDs start from 1

        plt.figure(figsize=(10, 8))
        plt.title(f'Graph with {num_clusters} Clusters (Euclidean Distance on Adjacency Rows)')
        
        # Use a spring layout for better visualization of graph structure
        pos = nx.spring_layout(G, seed=42) # For reproducible layout

        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500, alpha=0.9)
        nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.5, edge_color='gray')
        nx.draw_networkx_labels(G, pos, font_size=8, font_color='black')
        
        # Create a legend for clusters if desired (more complex, not implemented here for brevity)
        # For a simple legend, you might iterate through unique cluster IDs and plot a dummy point.

        plt.show()
        print(f"Nodes assigned to clusters: {clusters}")
        for i in range(1, num_clusters + 1):
            nodes_in_cluster = [idx for idx, c in enumerate(clusters) if c == i]
            print(f"Cluster {i}: {nodes_in_cluster}")
    else:
        print("\nNo specific number of clusters requested. Review the dendrogram to choose a cut-off point.")
        print("You can rerun the function with 'num_clusters' argument to visualize the graph with clusters.")

# --- Run the simulation ---

# Example 1: Generate a graph and just show the dendrogram
generate_random_graph_and_cluster(num_nodes=30, edge_probability=0.2, num_clusters=None)

# Example 2: Generate another graph and automatically extract 3 clusters
# You might need to adjust num_clusters based on the dendrogram structure
generate_random_graph_and_cluster(num_nodes=25, edge_probability=0.3, num_clusters=3)

# Example 3: A denser graph, might yield different clustering
generate_random_graph_and_cluster(num_nodes=20, edge_probability=0.5, num_clusters=2)