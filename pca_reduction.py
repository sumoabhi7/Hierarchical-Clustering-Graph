# pca_reduction.py
import numpy as np
from sklearn.decomposition import PCA
from typing import Tuple, Any

def train_pca(feature_vectors: np.ndarray, n_components=0.95) -> Tuple[PCA, np.ndarray]:
    n_samples, n_features = feature_vectors.shape
    if isinstance(n_components, int) and n_components > min(n_samples, n_features):
        n_components = min(n_samples, n_features)

    print(f"Training PCA with n_components={n_components}...")
    pca = PCA(n_components=n_components)
    reduced_features = pca.fit_transform(feature_vectors)
    print(f"PCA trained. Components: {pca.n_components_}.")
    return pca, reduced_features