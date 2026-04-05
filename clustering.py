# clustering.py
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from typing import Tuple, Dict

N_CLUSTERS = 20

def perform_hierarchical_clustering(features: np.ndarray) -> Tuple[AgglomerativeClustering, np.ndarray]:
    n_clusters = N_CLUSTERS
    if features.shape[0] < n_clusters:
        n_clusters = max(2, features.shape[0] // 2)
        print(f"Adjusted clustering to n_clusters={n_clusters} based on sample size.")

    print(f"Running AgglomerativeClustering with n_clusters={n_clusters}")
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    labels = model.fit_predict(features)
    print(f"Clustering complete: found {len(np.unique(labels))} labels.")
    return model, labels

def calculate_centroids(features: np.ndarray, labels: np.ndarray) -> Dict[int, np.ndarray]:
    print("Calculating centroids for clusters...")
    centroids = {}
    for lbl in np.unique(labels):
        centroids[int(lbl)] = np.mean(features[labels == lbl], axis=0)
    print(f"Calculated {len(centroids)} centroids.")
    return centroids