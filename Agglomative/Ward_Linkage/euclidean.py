import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import scipy.cluster.hierarchy as shc

# 1. Load and prepare data
data = pd.read_csv('synthetic_cc_data.csv')

# Drop customer ID column
X = data.drop('CUST_ID', axis=1)

# Handle missing values
X.fillna(X.mean(), inplace=True)  # Fill missing values with column means

# 2. Preprocess data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_normalized = normalize(X_scaled)

# 3. Dimensionality reduction with PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_normalized)
X_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])

# 4. Visualize dendrogram
plt.figure(figsize=(12, 8))
plt.title('Credit Card Customer Dendrogram (Ward Linkage)')
dend = shc.dendrogram(shc.linkage(X_normalized, method='ward'))
plt.axhline(y=6, color='r', linestyle='--')  # Suggested cutoff line
plt.xlabel('Customers')
plt.ylabel('Distance')
plt.show()

# 5. Perform clustering for different k values
for k in [2, 3, 4, 5]:  # Test different numbers of clusters
    # Create clustering model
    cluster = AgglomerativeClustering(n_clusters=k, affinity='euclidean', linkage='ward')
    cluster_labels = cluster.fit_predict(X_normalized)
    
    # Calculate silhouette score
    silhouette_avg = silhouette_score(X_normalized, cluster_labels)
    print(f"For n_clusters={k}, Silhouette Score: {silhouette_avg:.3f}")
    
    # Visualize clusters in PCA space
    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca['PC1'], X_pca['PC2'], c=cluster_labels, cmap='viridis', s=50)
    plt.title(f'Hierarchical Clustering (k={k})\nSilhouette Score: {silhouette_avg:.3f}')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.colorbar(label='Cluster')
    plt.grid(alpha=0.3)
    plt.show()
    
    # Add cluster labels to original data
    data[f'Cluster_k{k}'] = cluster_labels

# 6. Save results
data.to_csv('customer_clusters_results.csv', index=False)
print("Clustering results saved to 'customer_clusters_results.csv'")