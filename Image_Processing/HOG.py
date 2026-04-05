import numpy as np
import glob
from skimage.io import imread
from skimage.feature import hog
from scipy.spatial import distance
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

# read the images and store in a list
# Assumes images are in a folder named 'images' in the same directory as the script
images = [imread(file) for file in glob.glob("images/*.jpg")]

# number of images
n = len(images)

# show images
fig = plt.figure(figsize=(16, 8))
for i in range(n):
    fig.add_subplot(2, 4, i + 1)
    plt.imshow(images[i])
    plt.axis("off")
    plt.title(i)
plt.show()

fd_list = []
fig = plt.figure(figsize=(12, 12))
k = 0
for i in range(n):
    # Prepare parameters for the hog function
    hog_params = {
        'orientations': 9,
        'pixels_per_cell': (64, 64),
        'cells_per_block': (2, 2),
        'visualize': True
    }
    
    # Check if the image is a color image (3D). If so, add the channel_axis parameter.
    if images[i].ndim == 3:
        hog_params['channel_axis'] = -1
        
    # execute hog function with the correct parameters
    fd, hog_image = hog(images[i], **hog_params)
    
    # add the feature vector to the list
    fd_list.append(fd)
    
    # display original image
    fig.add_subplot(4, 4, k + 1)
    plt.imshow(images[i])
    plt.axis("off")
    plt.title(f"Original {i}")
    
    # display hog image
    fig.add_subplot(4, 4, k + 2)
    plt.imshow(hog_image)
    plt.axis("off")
    plt.title(f"HOG {i}")
    k += 2
plt.show()

distance_matrix = np.zeros((n, n))
for i in range(n):
    fd_i = fd_list[i]
    for k in range(i):
        fd_k = fd_list[k]
        # measure Jensen–Shannon distance between each feature vector
        # and add to the distance matrix
        distance_matrix[i, k] = distance.jensenshannon(fd_i, fd_k)

# symmetrize the matrix as distance matrix is symmetric
distance_matrix = np.maximum(distance_matrix, distance_matrix.transpose())

# The linkage function requires a condensed distance matrix (1D array)
# We convert the square distance matrix using squareform
cond_distance_matrix = squareform(distance_matrix)

# Perform hierarchical clustering
Z = linkage(cond_distance_matrix, method='ward')

# Plot the dendrogram
plt.figure(figsize=(12, 6))
dendrogram(Z, color_threshold=0.2, show_leaf_counts=True)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Image Index")
plt.ylabel("Distance")
plt.show()