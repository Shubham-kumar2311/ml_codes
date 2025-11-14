import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

X = np.array([
    [1, 2],
    [1.5, 1.8],
    [5, 8],
    [8, 8],
    [1, 0.6],
    [9, 11]
])

# 'single', 'complete', 'average', 'ward', 'centroid', 'median', or 'weighted'
Z = linkage(X, method='single')


plt.figure(figsize=(6, 4))
dendrogram(Z)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Data points")
plt.ylabel("Distance")
plt.show()


clusters = fcluster(Z, t=2, criterion='maxclust')
print("Cluster labels:", clusters)
