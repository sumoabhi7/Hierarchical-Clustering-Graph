import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import dendrogram, linkage

# --- Step 1: Prepare the Data ---
def prepare_image_data(image_path, size=(20, 20)):
    """
    Loads an image, converts it to RGB, resizes it, and flattens its pixel data
    into a 2D array for clustering.
    
    Args:
        image_path (str): The path to the image file.
        size (tuple): A tuple (width, height) to resize the image to. Smaller sizes
                      are better for a clear dendrogram.
    
    Returns:
        tuple: A tuple containing the flattened pixel data (np.ndarray) and
               the image size (tuple). Returns (None, None) on error.
    """
    try:
        img = Image.open(image_path)
        img = img.resize(size) 
        
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

# --- Step 2: Implement Divisive Clustering (fixed number of clusters) ---
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
    
    def get_max_diameter_cluster(current_clusters):
        """
        Helper function to find the cluster with the largest diameter.
        """
        max_diam = -1
        max_diam_cluster_idx = -1
        
        for i, cluster_indices in enumerate(current_clusters):
            if len(cluster_indices) > 1:
                sub_data = data[cluster_indices]
                distances = pdist(sub_data, metric='euclidean')
                diameter = np.max(distances) if distances.size > 0 else 0
                if diameter > max_diam:
                    max_diam = diameter
                    max_diam_cluster_idx = i
        return max_diam_cluster_idx, max_diam

    while len(clusters) < max_clusters:
        cluster_to_split_idx, _ = get_max_diameter_cluster(clusters)

        if cluster_to_split_idx == -1:
            break

        cluster_to_split_indices = clusters.pop(cluster_to_split_idx)
        
        # Split the cluster using K-means (k=2)
        kmeans = KMeans(n_clusters=2, random_state=0, n_init=10).fit(data[cluster_to_split_indices])
        labels = kmeans.labels_
        
        new_cluster1_indices = cluster_to_split_indices[labels == 0]
        new_cluster2_indices = cluster_to_split_indices[labels == 1]
        
        # Avoid splitting if K-means produced an empty cluster
        if len(new_cluster1_indices) == 0 or len(new_cluster2_indices) == 0:
            clusters.append(cluster_to_split_indices)
            break
        
        clusters.append(new_cluster1_indices)
        clusters.append(new_cluster2_indices)
        
    return clusters

# --- Step 3: Implement Divisive Clustering (full hierarchy) ---
def divisive_clustering_full_hierarchy(data):
    """
    Performs divisive hierarchical clustering until each pixel is its own cluster.
    
    Returns the full split history for dendrogram generation.
    """
    n_pixels = len(data)
    clusters = [np.arange(n_pixels)]
    dendrogram_data = []

    def get_max_diameter_cluster(current_clusters):
        max_diam = -1
        max_diam_cluster_idx = -1
        for i, cluster_indices in enumerate(current_clusters):
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
        cluster_to_split_idx, split_distance = get_max_diameter_cluster(clusters)
        
        if cluster_to_split_idx == -1:
            break
            
        cluster_to_split_indices = clusters.pop(cluster_to_split_idx)
        
        kmeans = KMeans(n_clusters=2, random_state=0, n_init=10).fit(data[cluster_to_split_indices])
        labels = kmeans.labels_
        
        new_cluster1_indices = cluster_to_split_indices[labels == 0]
        new_cluster2_indices = cluster_to_split_indices[labels == 1]
        
        # Avoid splitting if K-means produced an empty cluster
        if len(new_cluster1_indices) == 0 or len(new_cluster2_indices) == 0:
            clusters.append(cluster_to_split_indices)
            break

        clusters.append(new_cluster1_indices)
        clusters.append(new_cluster2_indices)

        dendrogram_data.append({
            'split_distance': split_distance,
            'child1_indices': new_cluster1_indices.tolist(),
            'child2_indices': new_cluster2_indices.tolist()
        })
    return dendrogram_data

# --- Step 4: Create a Linkage Matrix for SciPy ---
def create_scipy_linkage_matrix(dendrogram_data, n_pixels):
    """
    Converts the divisive split history into a linkage matrix compatible with SciPy's dendrogram.
    
    Args:
        dendrogram_data (list): The list of split history dictionaries.
        n_pixels (int): The total number of pixels.
        
    Returns:
        np.ndarray: The linkage matrix suitable for scipy.
    """
    if not dendrogram_data:
        return np.zeros((0, 4))

    linkage_matrix = np.zeros((n_pixels - 1, 4))
    
    next_cluster_id = n_pixels
    cluster_map = {frozenset([i]): i for i in range(n_pixels)}
    
    # Process the splits in reverse order to simulate merges
    for i, split_info in enumerate(reversed(dendrogram_data)):
        split_distance = split_info['split_distance']
        child1_indices = frozenset(split_info['child1_indices'])
        child2_indices = frozenset(split_info['child2_indices'])
        
        # Find the IDs of the two child clusters
        child1_id = cluster_map.get(child1_indices)
        if child1_id is None:
            child1_id = next_cluster_id
            cluster_map[child1_indices] = next_cluster_id
            next_cluster_id += 1
            
        child2_id = cluster_map.get(child2_indices)
        if child2_id is None:
            child2_id = next_cluster_id
            cluster_map[child2_indices] = next_cluster_id
            next_cluster_id += 1
        
        parent_indices = child1_indices.union(child2_indices)
        cluster_map[parent_indices] = next_cluster_id
        
        # Populate a row of the linkage matrix
        linkage_matrix[i, 0] = min(child1_id, child2_id)
        linkage_matrix[i, 1] = max(child1_id, child2_id)
        linkage_matrix[i, 2] = split_distance
        linkage_matrix[i, 3] = len(parent_indices)
        
        next_cluster_id += 1
        
    return linkage_matrix

# --- Step 5: Visualize Functions ---
def print_divisive_clustering_summary(pixel_data, final_clusters):
    """
    Generates and prints a summary of the final clusters.
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

def plot_divisive_dendrogram(dendrogram_data, n_pixels):
    """
    Creates and plots a dendrogram from divisive clustering split history
    using SciPy's standard dendrogram function.
    """
    linkage_matrix = create_scipy_linkage_matrix(dendrogram_data, n_pixels)
    
    plt.figure(figsize=(15, 8))
    dendrogram(linkage_matrix)
    
    # Invert the y-axis to make it look like a top-down, divisive dendrogram
    plt.gca().invert_yaxis()  
    
    plt.title('Divisive Clustering Dendrogram (Inverted Y-Axis)')
    plt.xlabel('Pixel Index')
    plt.ylabel('Distance')
    plt.show()

# --- Main Execution ---
if __name__ == "__main__":
    image_file = 'dummy_img.png'
    
    # Resize to a very small image for a readable dendrogram
    pixel_data, image_size = prepare_image_data(image_file, size=(10, 10))
    
    if pixel_data is not None:
        # Run divisive clustering to get a fixed number of clusters
        target_clusters = 5
        final_clusters = divisive_clustering(pixel_data, max_clusters=target_clusters)
        
        print(f"\nClustering to {target_clusters} clusters complete!\n")

        # Generate and print the summary
        print_divisive_clustering_summary(pixel_data, final_clusters)
        
        # Visualize the clustered image
        visualize_clusters(pixel_data, final_clusters, image_size)

        # Run the full hierarchy clustering for the dendrogram
        print("\nRunning full hierarchy clustering for dendrogram...")
        dendrogram_info = divisive_clustering_full_hierarchy(pixel_data)
        
        # Plot the dendrogram using the corrected function
        plot_divisive_dendrogram(dendrogram_info, len(pixel_data))