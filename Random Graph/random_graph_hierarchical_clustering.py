import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist, squareform

def plot_graph(G, title="Graph Visualization", node_colors=None, node_labels=True, edge_weights=False):
    """
    Helper function to plot a NetworkX graph.
    """
    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(G, seed=42) 

    edge_widths = None
    if edge_weights and nx.get_edge_attributes(G, 'weight'):
        edge_widths = [G[u][v]['weight'] for u, v in G.edges()]
        edge_widths = [w * 2 for w in edge_widths] # Make weighted edges more visible

    nx.draw_networkx_nodes(G, pos, node_color=node_colors if node_colors else 'skyblue',
                           node_size=600, alpha=0.9, linewidths=1, edgecolors='black')
    nx.draw_networkx_edges(G, pos, width=edge_widths if edge_widths else 0.5,
                           alpha=0.5, edge_color='gray')
    if node_labels:
        nx.draw_networkx_labels(G, pos, font_size=8, font_color='black')
    
    plt.title(title, fontsize=16)
    plt.axis('off') # Turn off axis
    plt.show()

def generate_random_graph_and_cluster_interactive():
    """
    Interactively generates a random graph, performs hierarchical clustering
    based on user input, and visualizes the results.
    """
    print("--- Interactive Hierarchical Clustering on Random Graphs ---")

    while True:
        try:
            num_nodes = int(input("Enter the number of nodes for the graph (e.g., 20-500): "))
            if 2 <= num_nodes <= 1000:
                break
            else:
                print("Please enter a number between 2 and 1000.")
        except ValueError:
            print("Invalid input. Please enter an integer.")

    while True:
        try:
            edge_probability = float(input("Enter the probability of an edge existing between any two nodes (0.0 to 1.0, e.g., 0.2): "))
            if 0.0 <= edge_probability <= 1.0:
                break
            else:
                print("Please enter a value between 0.0 and 1.0.")
        except ValueError:
            print("Invalid input. Please enter a floating-point number.")

    available_linkage_methods = ['ward', 'single', 'complete', 'average', 'weighted', 'centroid', 'median']
    print("\nAvailable Linkage Methods:")
    for i, method in enumerate(available_linkage_methods):
        print(f"  {i+1}. {method}")
    while True:
        try:
            linkage_choice = int(input(f"Choose a linkage method (1-{len(available_linkage_methods)}): "))
            if 1 <= linkage_choice <= len(available_linkage_methods):
                linkage_method = available_linkage_methods[linkage_choice - 1]
                break
            else:
                print("Invalid choice. Please enter a number from the list.")
        except ValueError:
            print("Invalid input. Please enter an integer.")

    if linkage_method == 'ward':
        distance_metric = 'euclidean'
        print(f"\n'Ward' linkage selected. Distance metric automatically set to 'euclidean'.")
    else:
        available_distance_metrics = ['euclidean', 'cityblock', 'cosine', 'correlation', 'jaccard', 'chebyshev']
        print("\nAvailable Distance Metrics:")
        for i, metric in enumerate(available_distance_metrics):
            print(f"  {i+1}. {metric}")
        while True:
            try:
                metric_choice = int(input(f"Choose a distance metric (1-{len(available_distance_metrics)}): "))
                if 1 <= metric_choice <= len(available_distance_metrics):
                    distance_metric = available_distance_metrics[metric_choice - 1]
                    break
                else:
                    print("Invalid choice. Please enter a number from the list.")
            except ValueError:
                print("Invalid input. Please enter an integer.")

    print(f"\nGenerating a random graph with {num_nodes} nodes, edge probability {edge_probability}.")
    print(f"Clustering with '{linkage_method}' linkage and '{distance_metric}' distance.")

    adj_matrix = np.zeros((num_nodes, num_nodes))

    np.random.seed(42)
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if np.random.rand() < edge_probability:
                adj_matrix[i, j] = 1
                adj_matrix[j, i] = 1

    G = nx.from_numpy_array(adj_matrix)
    print(f"Graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

    if not nx.is_connected(G):
        print("Warning: The generated graph is not connected. Clustering might produce disconnected components.")
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
        num_nodes = G.number_of_nodes()
        print(f"Proceeding with the largest connected component: {num_nodes} nodes.")
        adj_matrix = nx.to_numpy_array(G) 
    if num_nodes == 0:
        print("Error: No connected nodes found in the graph. Cannot perform clustering.")
        return

    plot_graph(G, title=f"Initial Random Graph (Nodes: {num_nodes}, Edges: {G.number_of_edges()})")

    node_features = adj_matrix
    
    print("\nCalculating pairwise distances...")
    dissimilarity_matrix = pdist(node_features, metric=distance_metric)

    print(f"Performing hierarchical clustering with '{linkage_method}' linkage...")
    Z = linkage(dissimilarity_matrix, method=linkage_method)

    plt.figure(figsize=(15, max(8, num_nodes * 0.3)))
    plt.title(f'Hierarchical Clustering Dendrogram (Method: {linkage_method}, Metric: {distance_metric})')
    plt.xlabel('Node Index')
    plt.ylabel('Distance')
    dendrogram(
        Z,
        labels=np.arange(num_nodes), # Using numerical indices from 0 to num_nodes-1
        leaf_rotation=90.,
        leaf_font_size=8.,
        show_contracted=True
    )
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

    while True:
        extract_clusters_input = input("\nDo you want to extract and visualize clusters on the graph? (yes/no): ").lower()
        if extract_clusters_input in ['yes', 'no']:
            break
        else:
            print("Invalid input. Please enter 'yes' or 'no'.")

    if extract_clusters_input == 'yes':
        while True:
            try:
                num_clusters_to_extract = int(input(f"Enter the desired number of clusters (e.g., 2 to {num_nodes}): "))
                if 1 <= num_clusters_to_extract <= num_nodes:
                    break
                else:
                    print(f"Please enter a number between 1 and {num_nodes}.")
            except ValueError:
                print("Invalid input. Please enter an integer.")
        
        print(f"\nExtracting {num_clusters_to_extract} clusters...")
        clusters = fcluster(Z, num_clusters_to_extract, criterion='maxclust')
        
        node_to_cluster_id = {node: cluster_id for node, cluster_id in zip(G.nodes(), clusters)}

        cmap = plt.cm.get_cmap('viridis', num_clusters_to_extract)
        node_colors = [cmap(node_to_cluster_id[node] - 1) for node in G.nodes()] 

        plot_graph(G, title=f'Graph with {num_clusters_to_extract} Clusters (Method: {linkage_method}, Metric: {distance_metric})',
                   node_colors=node_colors)
        
        handles = [plt.Line2D([0], [0], marker='o', color='w', label=f'Cluster {i}',
                              markerfacecolor=cmap(i-1), markersize=10) for i in range(1, num_clusters_to_extract + 1)]
        plt.legend(handles=handles, title="Clusters", loc='best')
        plt.show() 

        print(f"\nNodes assigned to clusters:")
        for i in range(1, num_clusters_to_extract + 1):
            nodes_in_cluster = [node for node, c_id in node_to_cluster_id.items() if c_id == i]
            print(f"  Cluster {i}: {sorted(nodes_in_cluster)}")
    else:
        print("No clusters extracted and visualized on the graph.")

generate_random_graph_and_cluster_interactive()