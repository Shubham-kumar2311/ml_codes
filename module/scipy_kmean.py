from scipy.cluster.vq import kmeans, vq, whiten
import numpy as np

X = np.array([
    [1, 2],
    [1.5, 1.8],
    [5, 8],
    [8, 8],
    [1, 0.6],
    [9, 11]
])


X_white = whiten(X) # var = 1

# Perform K-Means clustering
centroids, distortion = kmeans(X_white, 2)

# Assign each point to a cluster
cluster_labels, _ = vq(X_white, centroids)

print("Centroids:\n", centroids)
print("Cluster labels:", cluster_labels)
