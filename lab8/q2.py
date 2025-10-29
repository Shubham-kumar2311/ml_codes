import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

# ---------- (a) Read the dataset ----------
data = pd.read_csv("Mall_Customers.csv")
X = data[['Annual Income (k$)', 'Spending Score (1-100)']].values


# ---------- Helper: Euclidean distance ----------
def euclidean_distance(a, b):
    return sqrt(np.sum((a - b) ** 2))


# ---------- Distance matrix ----------
def distance_matrix(X):
    n = len(X)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist[i][j] = euclidean_distance(X[i], X[j])
    return dist


# ---------- Hierarchical clustering ----------
def hierarchical_clustering(X, linkage='single', n_clusters=5):
    clusters = [[i] for i in range(len(X))]  # each point as its own cluster
    dist = distance_matrix(X)

    while len(clusters) > n_clusters:
        # find the closest pair of clusters
        min_dist = float('inf')
        c1, c2 = -1, -1
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                if linkage == 'single':
                    d = min([dist[p1][p2] for p1 in clusters[i] for p2 in clusters[j]])
                elif linkage == 'complete':
                    d = max([dist[p1][p2] for p1 in clusters[i] for p2 in clusters[j]])
                elif linkage == 'centroid':
                    centroid1 = np.mean(X[clusters[i]], axis=0)
                    centroid2 = np.mean(X[clusters[j]], axis=0)
                    d = euclidean_distance(centroid1, centroid2)
                if d < min_dist:
                    min_dist = d
                    c1, c2 = i, j

        # merge the closest clusters
        new_cluster = clusters[c1] + clusters[c2]
        clusters.pop(max(c1, c2))
        clusters.pop(min(c1, c2))
        clusters.append(new_cluster)

    # assign cluster labels
    labels = np.zeros(len(X))
    for cluster_id, cluster_points in enumerate(clusters):
        for p in cluster_points:
            labels[p] = cluster_id

    return labels


# ---------- (b) Run for all 3 linkages ----------
linkage_types = ['single', 'complete', 'centroid']

for link in linkage_types:
    labels = hierarchical_clustering(X, linkage=link, n_clusters=5)

    plt.figure(figsize=(6, 5))
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='rainbow')
    plt.title(f'Hierarchical Clustering ({link}-linkage)')
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.show()

    print(f"Linkage: {link}")
    print(f"Clusters formed: {len(set(labels))}\n")
