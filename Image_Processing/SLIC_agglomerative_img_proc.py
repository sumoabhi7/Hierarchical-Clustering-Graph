from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from skimage.segmentation import slic # For superpixel segmentation
from skimage.color import rgb2lab # For Lab color space
from skimage.util import img_as_float # To convert image to float for skimage
from sklearn.preprocessing import StandardScaler # For feature scaling
import time # To measure execution time

# -----------------------
# Step 1: Load and preprocess image
# -----------------------
image_path = r"dummy_img.png"  # IMPORTANT: Change to your image path (e.g., "my_image.jpg")
try:
    img = Image.open(image_path).convert('RGB').resize((200, 200))
except FileNotFoundError:
    print(f"Error: Image file not found at {image_path}. Please check the path.")
    print("Creating a dummy image for demonstration...")
    # Create a simple dummy image if not found
    dummy_img_array = np.zeros((200, 200, 3), dtype=np.uint8)
    dummy_img_array[0:100, 0:100] = [255, 0, 0] # Red top-left
    dummy_img_array[0:100, 100:200] = [0, 255, 0] # Green top-right
    dummy_img_array[100:200, 0:100] = [0, 0, 255] # Blue bottom-left
    dummy_img_array[100:200, 100:200] = [255, 255, 0] # Yellow bottom-right
    img = Image.fromarray(dummy_img_array)


img_array_rgb = np.array(img)  # Shape: (200, 200, 3)
print(f"Original image loaded. Resolution: {img_array_rgb.shape[0]}x{img_array_rgb.shape[1]}")

# Convert image to float for skimage processing
img_float = img_as_float(img_array_rgb)

# Convert to Lab color space for better perceptual distance
img_lab = rgb2lab(img_float)
print(f"Image converted to Lab color space. Shape: {img_lab.shape}")

# Reshape color pixels for clustering (L, a, b values)
pixels_lab = img_lab.reshape(-1, 3) # (40000, 3)

# Generate spatial coordinates for each pixel
h, w, _ = img_array_rgb.shape
coords = np.array([(i, j) for i in range(h) for j in range(w)]) # (40000, 2)

# -----------------------
# Step 2: Superpixel Segmentation (Optimization for Speed)
# -----------------------
print("\nPerforming superpixel segmentation (SLIC)...")
start_time = time.time()
# n_segments: desired number of superpixels
# compactness: balances color proximity and space proximity
# multichannel=True is for color images
slic_labels = slic(img_float, n_segments=500, compactness=10, channel_axis=-1, sigma=1, start_label=1)
num_superpixels = slic_labels.max() # Actual number of superpixels might be slightly different than n_segments
slic_labels_flat = slic_labels.reshape(-1) # Flatten superpixel labels to match pixel_lab/coords shape
end_time = time.time()
print(f"Superpixel segmentation complete. Found {num_superpixels} superpixels in {end_time - start_time:.2f} seconds.")

# -----------------------
# Step 3: Feature Extraction for Superpixels (Accuracy)
# -----------------------
print("Extracting features for each superpixel (Lab color + Spatial Coords)...")
superpixel_features = np.zeros((num_superpixels, 5)) # 3 for Lab, 2 for x,y

for i in range(1, num_superpixels + 1): # SLIC labels are 1-indexed by default
    mask = (slic_labels_flat == i)
    
    # Average Lab color for the superpixel
    avg_lab = pixels_lab[mask].mean(axis=0)
    
    # Average spatial coordinates for the superpixel
    avg_coords = coords[mask].mean(axis=0)
    
    superpixel_features[i-1] = np.concatenate((avg_lab, avg_coords))

print(f"Superpixel features shape: {superpixel_features.shape}")

# Scale features for better distance calculation (Accuracy)
scaler = StandardScaler()
scaled_superpixel_features = scaler.fit_transform(superpixel_features)
print("Superpixel features scaled.")

# -----------------------
# Step 4: Create dendrogram (on Superpixels)
# -----------------------
# The dendrogram is now much faster as it's built on a reduced number of data points
print("\nCreating dendrogram from superpixel features (this should be faster)...")
start_time = time.time()
linked = linkage(scaled_superpixel_features, method='ward', metric='euclidean') # 'ward' implicitly uses 'euclidean'
end_time = time.time()
print(f"Dendrogram linkage calculated in {end_time - start_time:.2f} seconds.")

plt.figure(figsize=(12, 6)) # Adjusted figsize for potentially wider dendrogram
dendrogram(linked,
           orientation='top',
           distance_sort='descending',
           show_leaf_counts=True,
           leaf_rotation=90., # Rotate labels for readability
           leaf_font_size=8.)
plt.title(f"Dendrogram (Superpixels - {num_superpixels} points)")
plt.xlabel("Superpixel Index")
plt.ylabel("Euclidean Distance (Scaled Features)")
plt.tight_layout() # Adjust layout to prevent labels from overlapping
plt.show()

# -----------------------
# Step 5: Apply hierarchical clustering (on Superpixels)
# -----------------------
print("\nApplying hierarchical clustering on superpixel features...")
n_clusters = 5  # You can adjust this or decide based on the dendrogram
user_n_clusters = input(f"Enter desired number of final clusters (e.g., {n_clusters}, or based on dendrogram): ")
try:
    n_clusters = int(user_n_clusters)
    if not (1 <= n_clusters <= num_superpixels):
        print(f"Invalid number of clusters. Defaulting to {n_clusters} or {min(5, num_superpixels)}.")
        n_clusters = min(5, num_superpixels) # Ensure default is reasonable
except ValueError:
    print(f"Invalid input. Defaulting to {n_clusters} or {min(5, num_superpixels)} clusters.")
    n_clusters = min(5, num_superpixels)

start_time = time.time()
clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward', metric='euclidean') # Metric explicitly set
superpixel_labels = clustering.fit_predict(scaled_superpixel_features)
end_time = time.time()
print(f"Clustering complete. Processed {num_superpixels} superpixels in {end_time - start_time:.2f} seconds.")

# -----------------------
# Step 6: Map Cluster Labels back to Original Pixels
# -----------------------
# Create an array to hold the final cluster label for each original pixel
final_pixel_labels = np.zeros(h * w, dtype=int)
for i in range(1, num_superpixels + 1):
    # Get the cluster label assigned to this superpixel
    cluster_id_for_superpixel = superpixel_labels[i-1]
    
    # Assign this cluster label to all original pixels belonging to this superpixel
    final_pixel_labels[slic_labels_flat == i] = cluster_id_for_superpixel

# Reshape back to image dimensions
final_pixel_labels_reshaped = final_pixel_labels.reshape(h, w)

# -----------------------
# Step 7: Analyze Clusters & Create Segmented Image
# -----------------------
print("\n--- Cluster Analysis ---")
unique_final_labels, counts = np.unique(final_pixel_labels, return_counts=True)
print("\nCluster Pixel Counts:")
for label, count in zip(unique_final_labels, counts):
    print(f"  Cluster {label}: {count} pixels")

# Calculate average color of each final cluster (using original RGB values)
centroid_colors_rgb = np.zeros((n_clusters, 3))
for i in range(n_clusters):
    # Find all original RGB pixels belonging to the current cluster
    cluster_rgb_pixels = img_array_rgb.reshape(-1, 3)[final_pixel_labels == i]
    
    # Calculate the average color
    if len(cluster_rgb_pixels) > 0: # Ensure cluster is not empty
        average_color = cluster_rgb_pixels.mean(axis=0)
        centroid_colors_rgb[i] = average_color
    else:
        # Handle empty clusters (shouldn't happen if n_clusters is reasonable)
        centroid_colors_rgb[i] = [0, 0, 0] # Default to black
    
    print(f"  Cluster {i} Average Color (R,G,B): {centroid_colors_rgb[i].astype(int)}")

# Create the segmented image using the average colors
segmented_img_display = centroid_colors_rgb[final_pixel_labels_reshaped].astype(np.uint8)

# -----------------------
# Step 8: Visualize Results
# -----------------------
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img_array_rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(segmented_img_display)
plt.title(f"Segmented Image ({n_clusters} Clusters)")
plt.axis("off")

plt.show()

print("\nProcessing complete!")