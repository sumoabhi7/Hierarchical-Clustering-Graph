# diana.py
import numpy as np
from scipy.spatial.distance import cdist

class diana:
    def __init__(self, data, number_clusters=2, metric='euclidean'):
        self.data = np.array(data)
        self.number_clusters = number_clusters
        self.metric = metric
        self.clusters = []

    def process(self):
        # Start with all data in one cluster
        self.clusters = [list(range(len(self.data)))]
        
        # Keep splitting until we have enough clusters
        while len(self.clusters) < self.number_clusters:
            # Find the largest cluster to split
            largest_cluster_idx = np.argmax([len(c) for c in self.clusters])
            cluster_to_split = self.clusters.pop(largest_cluster_idx)
            
            # If cluster too small to split, stop
            if len(cluster_to_split) <= 1:
                self.clusters.append(cluster_to_split)
                break

            # Simple split: pick the point furthest from the centroid to start new cluster
            cluster_points = self.data[cluster_to_split]
            centroid = np.mean(cluster_points, axis=0, keepdims=True)
            distances = cdist(cluster_points, centroid, metric=self.metric).flatten()
            idx_furthest = np.argmax(distances)
            new_cluster = [cluster_to_split.pop(idx_furthest)]
            
            # Move other points closer to new cluster
            for idx in cluster_to_split[:]:
                point = self.data[idx]
                dist_to_new = cdist([point], self.data[new_cluster], metric=self.metric).mean()
                dist_to_old = cdist([point], self.data[cluster_to_split], metric=self.metric).mean() if cluster_to_split else float('inf')
                if dist_to_new < dist_to_old:
                    cluster_to_split.remove(idx)
                    new_cluster.append(idx)

            self.clusters.append(cluster_to_split)
            self.clusters.append(new_cluster)

    def get_clusters(self):
        return self.clusters
