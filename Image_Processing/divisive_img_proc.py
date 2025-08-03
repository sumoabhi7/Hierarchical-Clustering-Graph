'''import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import dendrogram, linkage

# --- Step 1: Prepare the Data ---
def prepare_image_data(image_path):
    """Loads an image and flattens its pixel data into a 2D array."""
    try:
        img = Image.open(image_path)
        # REDUCED SIZE FOR FEWER NODES (10x10 = 100 pixels)
        img = img.resize((20, 20))
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        img_array = np.array(img)

        # Reshape the 3D array (height, width, RGB) into a 2D array (pixels, RGB)
        pixel_data = img_array.reshape(-1, 3)
        return pixel_data, img.size

    except FileNotFoundError:
        print(f"Error: The image file '{image_path}' was not found.")
        return None, None

# --- Step 2: Implement Divisive Clustering ---
def divisive_clustering(data, max_clusters):
    """
    Performs divisive hierarchical clustering using a recursive approach.
    Splitting is done using K-means with k=2.
    
    Returns a list of final clusters and a list of split events.
    """
    clusters = [np.arange(len(data))]
    n_pixels = len(data)
    dendrogram_data = []

    def get_max_diameter_cluster():
        """Find the cluster with the largest diameter (most dissimilar points)."""
        max_diam = -1
        max_diam_cluster_idx = -1
        
        for i, cluster_indices in enumerate(clusters):
            if len(cluster_indices) > 1:
                sub_data = data[cluster_indices]
                distances = pdist(sub_data, metric='euclidean')
                diameter = np.max(distances) if distances.size > 0 else 0
                if diameter > max_diam:
                    max_diam = diameter
                    max_diam_cluster_idx = i
        return max_diam_cluster_idx, max_diam

    while len(clusters) < max_clusters:
        cluster_to_split_idx, split_distance = get_max_diameter_cluster()

        if cluster_to_split_idx == -1:
            break

        cluster_to_split_indices = clusters.pop(cluster_to_split_idx)
        
        kmeans = KMeans(n_clusters=2, random_state=0, n_init=10).fit(data[cluster_to_split_indices])
        labels = kmeans.labels_
        
        new_cluster1_indices = cluster_to_split_indices[labels == 0]
        new_cluster2_indices = cluster_to_split_indices[labels == 1]
        
        clusters.append(new_cluster1_indices)
        clusters.append(new_cluster2_indices)

        dendrogram_data.append({
            'split_indices': cluster_to_split_indices.tolist(),
            'split_distance': split_distance,
            'child1_indices': new_cluster1_indices.tolist(),
            'child2_indices': new_cluster2_indices.tolist()
        })
        
        print(f"Splitting cluster of size {len(cluster_to_split_indices)} into two new clusters. Total clusters: {len(clusters)}")
        
    return clusters, dendrogram_data

# --- Step 3: Visualize the Clusters on the Image ---
def visualize_clusters(pixel_data, clusters, image_size):
    """
    Creates a new image where each cluster is assigned a mean color.
    """
    clustered_image_data = np.zeros_like(pixel_data, dtype=np.uint8)
    
    for cluster_indices in clusters:
        if len(cluster_indices) > 0:
            cluster_pixels = pixel_data[cluster_indices]
            mean_color = np.mean(cluster_pixels, axis=0).astype(np.uint8)
            clustered_image_data[cluster_indices] = mean_color
    
    clustered_image = clustered_image_data.reshape(image_size[0], image_size[1], 3)
    
    plt.figure(figsize=(8, 8))
    plt.imshow(clustered_image)
    plt.title(f'Image with {len(clusters)} Clusters')
    plt.axis('off')
    plt.show()

# --- Main Execution ---
if __name__ == "__main__":
    image_file = 'dummy_image.png'
    
    # Step 1: Prepare the Data
    pixel_data, image_size = prepare_image_data(image_file)
    if pixel_data is not None:
        print(f"Image data prepared. Total pixels: {len(pixel_data)}")

        # Step 2: Perform Divisive Clustering
        target_clusters = 500
        final_clusters, _ = divisive_clustering(pixel_data, target_clusters)
        
        print(f"\nDivisive clustering completed. Final number of clusters: {len(final_clusters)}")
        
        # --- NEW Step 3: Visualize the Dendrogram (The Correct Way) ---
        # 1. Get the centroids of the final clusters
        cluster_centroids = [np.mean(pixel_data[c], axis=0) for c in final_clusters if len(c) > 0]
        
        # 2. Use SciPy's standard linkage function on these centroids
        linkage_matrix = linkage(cluster_centroids, method='ward')
        
        # 3. Plot the dendrogram
        plt.figure(figsize=(15, 8))
        dendrogram(linkage_matrix, labels=None)
        plt.title('Dendrogram of Final Clusters (using Ward linkage)')
        plt.xlabel('Cluster Index')
        plt.ylabel('Distance')
        plt.show()

        # Step 4: Visualize the Clustered Image
        visualize_clusters(pixel_data, final_clusters, image_size)'''


import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import dendrogram, linkage

# --- Step 1: Prepare the Data ---
def prepare_image_data(image_path):
    """
    Loads an image, converts it to RGB, resizes it, and flattens its pixel data
    into a 2D array for clustering.
    """
    try:
        img = Image.open(image_path)
        img_original_size = img.size
        # Resize for faster computation and a clearer dendrogram example
        img = img.resize((200, 200)) 
        
        # Ensure the image is in RGB format, discarding the alpha channel if present
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        
        img_array = np.array(img)
        
        # Reshape the 3D array (height, width, RGB) into a 2D array (pixels, RGB)
        pixel_data = img_array.reshape(-1, 3)
        print(f"Image converted to RGB. Shape: {img_array.shape}, Pixels: {pixel_data.shape}")
        
        return pixel_data, img.size

    except FileNotFoundError:
        print(f"Error: The image file '{image_path}' was not found.")
        return None, None

# --- Step 2: Implement Divisive Clustering ---
def divisive_clustering(data, max_clusters):
    """
    Performs divisive hierarchical clustering using a top-down approach.
    Splitting is done using K-means with k=2.
    
    Args:
        data (np.ndarray): The flattened pixel data.
        max_clusters (int): The number of final clusters to stop at.
        
    Returns:
        list: A list of arrays, where each array contains the pixel indices
              for a final cluster.
    """
    clusters = [np.arange(len(data))]
    dendrogram_data = []

    def get_max_diameter_cluster():
        """
        Helper function to find the cluster with the largest diameter (most dissimilar points).
        """
        max_diam = -1
        max_diam_cluster_idx = -1
        
        for i, cluster_indices in enumerate(clusters):
            if len(cluster_indices) > 1:
                sub_data = data[cluster_indices]
                distances = pdist(sub_data, metric='euclidean')
                diameter = np.max(distances) if distances.size > 0 else 0
                if diameter > max_diam:
                    max_diam = diameter
                    max_diam_cluster_idx = i
        return max_diam_cluster_idx, max_diam

    while len(clusters) < max_clusters:
        cluster_to_split_idx, split_distance = get_max_diameter_cluster()

        if cluster_to_split_idx == -1:
            break

        cluster_to_split_indices = clusters.pop(cluster_to_split_idx)
        
        # Split the cluster using K-means (k=2)
        kmeans = KMeans(n_clusters=2, random_state=0, n_init=10).fit(data[cluster_to_split_indices])
        labels = kmeans.labels_
        
        new_cluster1_indices = cluster_to_split_indices[labels == 0]
        new_cluster2_indices = cluster_to_split_indices[labels == 1]
        
        clusters.append(new_cluster1_indices)
        clusters.append(new_cluster2_indices)

        dendrogram_data.append({
            'split_indices': cluster_to_split_indices.tolist(),
            'split_distance': split_distance,
            'child1_indices': new_cluster1_indices.tolist(),
            'child2_indices': new_cluster2_indices.tolist()
        })
        
    return clusters, dendrogram_data

# --- Step 3: Generate and Print Summary Output ---
def print_divisive_clustering_summary(pixel_data, final_clusters):
    """
    Generates and prints a summary of the final clusters from divisive clustering,
    including pixel counts and average cluster colors.
    
    Args:
        pixel_data (np.ndarray): The original flattened pixel data.
        final_clusters (list): A list of arrays, where each array contains the
                               pixel indices for a final cluster.
    """
    print("\n--- Cluster Pixel Counts ---")
    for i, cluster_indices in enumerate(final_clusters):
        print(f"Cluster {i} : {len(cluster_indices)} pixels")
    
    print("\n--- Average Cluster Colors (Centroids) ---")
    for i, cluster_indices in enumerate(final_clusters):
        if len(cluster_indices) > 0:
            cluster_pixels = pixel_data[cluster_indices]
            mean_color = np.mean(cluster_pixels, axis=0).astype(np.uint8)
            print(f"Cluster {i} Average Color (R,G,B): {mean_color}")
        else:
            print(f"Cluster {i} Average Color (R,G,B): [0 0 0] (empty)")

# --- Step 4: Visualize the Clustered Image ---
def visualize_clusters(pixel_data, clusters, image_size):
    """
    Creates a new image where each cluster is assigned a mean color.
    """
    clustered_image_data = np.zeros_like(pixel_data, dtype=np.uint8)
    
    for cluster_indices in clusters:
        if len(cluster_indices) > 0:
            cluster_pixels = pixel_data[cluster_indices]
            mean_color = np.mean(cluster_pixels, axis=0).astype(np.uint8)
            clustered_image_data[cluster_indices] = mean_color
    
    clustered_image = clustered_image_data.reshape(image_size[0], image_size[1], 3)
    
    plt.figure(figsize=(8, 8))
    plt.imshow(clustered_image)
    plt.title(f'Image with {len(clusters)} Clusters')
    plt.axis('off')
    plt.show()

def divisive_clustering_full_hierarchy(data):
    """
    Performs divisive hierarchical clustering until each pixel is its own cluster.
    
    Returns the full split history for dendrogram generation.
    """
    n_pixels = len(data)
    clusters = [np.arange(n_pixels)]
    dendrogram_data = []

    def get_max_diameter_cluster():
        # (This helper function is the same as before)
        max_diam = -1
        max_diam_cluster_idx = -1
        for i, cluster_indices in enumerate(clusters):
            if len(cluster_indices) > 1:
                sub_data = data[cluster_indices]
                distances = pdist(sub_data, metric='euclidean')
                diameter = np.max(distances) if distances.size > 0 else 0
                if diameter > max_diam:
                    max_diam = diameter
                    max_diam_cluster_idx = i
        return max_diam_cluster_idx, max_diam

    # The loop runs until the number of clusters equals the number of pixels
    while len(clusters) < n_pixels:
        cluster_to_split_idx, split_distance = get_max_diameter_cluster()
        if cluster_to_split_idx == -1:
            break
        cluster_to_split_indices = clusters.pop(cluster_to_split_idx)
        
        kmeans = KMeans(n_clusters=2, random_state=0, n_init=10).fit(data[cluster_to_split_indices])
        labels = kmeans.labels_
        
        new_cluster1_indices = cluster_to_split_indices[labels == 0]
        new_cluster2_indices = cluster_to_split_indices[labels == 1]
        
        clusters.append(new_cluster1_indices)
        clusters.append(new_cluster2_indices)

        dendrogram_data.append({
            'split_indices': cluster_to_split_indices.tolist(),
            'split_distance': split_distance,
            'child1_indices': new_cluster1_indices.tolist(),
            'child2_indices': new_cluster2_indices.tolist()
        })
    return dendrogram_data

def create_scipy_linkage_matrix(dendrogram_data, n_pixels):
    """
    Converts the full divisive split history into a linkage matrix compatible with SciPy.
    
    This function simulates the agglomerative (bottom-up) process in reverse.
    """
    if not dendrogram_data:
        return np.zeros((0, 4))
    
    linkage_matrix = np.zeros((n_pixels - 1, 4))
    
    # Initialize the ID mapping with all singletons
    node_id_map = {tuple([i]): i for i in range(n_pixels)}
    next_node_id = n_pixels
    
    # Process splits in reverse order to simulate merges
    for i, split_info in enumerate(reversed(dendrogram_data)):
        child1_indices = tuple(sorted(split_info['child1_indices']))
        child2_indices = tuple(sorted(split_info['child2_indices']))
        
        child1_id = node_id_map.get(child1_indices)
        child2_id = node_id_map.get(child2_indices)
        
        # If a child ID is missing, it's a new cluster being formed by this split.
        if child1_id is None:
            child1_id = next_node_id
            node_id_map[child1_indices] = next_node_id
            next_node_id += 1
        if child2_id is None:
            child2_id = next_node_id
            node_id_map[child2_indices] = next_node_id
            next_node_id += 1

        # The parent of this split will be a new merged node
        parent_indices = tuple(sorted(split_info['split_indices']))
        node_id_map[parent_indices] = next_node_id
        
        # Populate a row of the linkage matrix
        linkage_matrix[n_pixels - 2 - i, 0] = min(child1_id, child2_id)
        linkage_matrix[n_pixels - 2 - i, 1] = max(child1_id, child2_id)
        linkage_matrix[n_pixels - 2 - i, 2] = split_info['split_distance']
        linkage_matrix[n_pixels - 2 - i, 3] = len(parent_indices)
        
        next_node_id += 1
        
    return linkage_matrix

def plot_divisive_dendrogram(dendrogram_data, n_pixels):
    """
    Creates and plots a dendrogram from divisive clustering split history
    using SciPy's standard dendrogram function.
    """
    linkage_matrix = create_scipy_linkage_matrix(dendrogram_data, n_pixels)
    
    # Create labels for the x-axis (from 0 to n_pixels-1)
    labels = np.arange(n_pixels).tolist()
    
    plt.figure(figsize=(15, 8))
    dendrogram(linkage_matrix, labels=labels)
    plt.title('Divisive Clustering Dendrogram for Each Pixel')
    plt.xlabel('Pixel Index')
    plt.ylabel('Distance')
    plt.show()

# --- Main Execution ---
if __name__ == "__main__":
    image_file = 'dummy_img.png'
    
    # Step 1: Prepare the Data
    pixel_data, image_size = prepare_image_data(image_file)
    if pixel_data is not None:
        # Run divisive clustering to get a fixed number of clusters
        target_clusters = 5
        final_clusters, _ = divisive_clustering(pixel_data, max_clusters=target_clusters)
        
        print(f"\nClustering complete!\n")

        # Step 2: Generate and print the summary
        print_divisive_clustering_summary(pixel_data, final_clusters)
        
        # Step 3: Visualize the clustered image
        visualize_clusters(pixel_data, final_clusters, image_size)

        # 1. Run the divisive clustering all the way to completion (down to each pixel)
        print("\nRunning full hierarchy clustering for dendrogram...")
        dendrogram_info = divisive_clustering_full_hierarchy(pixel_data)
        
        # 2. Call the new plotting function to generate the dendrogram
        plot_divisive_dendrogram(dendrogram_info, len(pixel_data))