import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

data = pd.read_csv("Mall_Customers.csv")
X = data[['Annual Income (k$)', 'Spending Score (1-100)']].values

def euclidean_distance(a, b):
    return sqrt(np.sum((a - b) ** 2))

def cluster_variance(points):
    if len(points) == 0:
        return 0
    centroid = np.mean(points, axis=0)
    return np.sum(np.linalg.norm(points - centroid, axis=1)**2)

def divisive_clustering(X, n_clusters=5):
    clusters = [np.arange(len(X))]
    
    while len(clusters) < n_clusters:
        # choose the cluster with largest variance to split
        max_var = -1
        cluster_to_split = -1
        for idx, cluster in enumerate(clusters):
            var = cluster_variance(X[cluster])
            if var > max_var:
                max_var = var
                cluster_to_split = idx
        
        points = X[clusters[cluster_to_split]]
        # split using 2-means (k=2) within the selected cluster
        centroid1 = points[0]
        centroid2 = points[len(points)//2]
        labels = np.zeros(len(points))
        for i, p in enumerate(points):
            d1 = euclidean_distance(p, centroid1)
            d2 = euclidean_distance(p, centroid2)
            labels[i] = 0 if d1 < d2 else 1
        cluster0 = clusters[cluster_to_split][labels==0]
        cluster1 = clusters[cluster_to_split][labels==1]
        
        # replace old cluster with the two new clusters
        clusters.pop(cluster_to_split)
        clusters.append(cluster0)
        clusters.append(cluster1)
    
    # assign labels
    final_labels = np.zeros(len(X))
    for cluster_id, cluster_points in enumerate(clusters):
        for p in cluster_points:
            final_labels[p] = cluster_id
    return final_labels

labels = divisive_clustering(X, n_clusters=5)

plt.figure(figsize=(6, 5))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='rainbow')
plt.title('Divisive Hierarchical Clustering')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.show()

print(f"Clusters formed: {len(set(labels))}")
