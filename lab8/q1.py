import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

# ---------- (a) Read the dataset ----------
data = pd.read_csv("Mall_Customers.csv")
X = data[['Annual Income (k$)', 'Spending Score (1-100)']].values


# ---------- Helper functions ----------
def euclidean_distance(a, b):
    return sqrt(np.sum((a - b) ** 2))


def region_query(X, point_idx, eps):
    neighbors = []
    for i in range(len(X)):
        if euclidean_distance(X[point_idx], X[i]) <= eps:
            neighbors.append(i)
    return neighbors


def expand_cluster(X, labels, point_idx, cluster_id, eps, min_pts):
    neighbors = region_query(X, point_idx, eps)
    if len(neighbors) < min_pts:
        labels[point_idx] = -1  # noise
        return False
    else:
        labels[point_idx] = cluster_id
        i = 0
        while i < len(neighbors):
            n = neighbors[i]
            if labels[n] == -1:
                labels[n] = cluster_id
            elif labels[n] == 0:
                labels[n] = cluster_id
                new_neighbors = region_query(X, n, eps)
                if len(new_neighbors) >= min_pts:
                    neighbors += new_neighbors
            i += 1
        return True


def dbscan(X, eps, min_pts):
    cluster_id = 0
    labels = np.zeros(X.shape[0])
    # print(labels)
    for i in range(len(X)):
        if labels[i] == 0:
            if expand_cluster(X, labels, i, cluster_id + 1, eps, min_pts):
                cluster_id += 1
    return labels


# print(dbscan(X,2,4))

# ---------- (b) Hyperparameter tuning ----------
min_pts_values = range(4, 11)
eps_values = [2, 2.5, 3, 3.5, 4]

best_config = None
best_clusters = 0

for min_pts in min_pts_values:
    for eps in eps_values:
        labels = dbscan(X, eps, min_pts)
        clusters = len(set(labels)) - (1 if -1 in labels else 0)
        print(f"min_pts={min_pts}, eps={eps}, clusters={clusters}")
        if clusters > best_clusters:
            best_clusters = clusters
            best_config = (min_pts, eps)

print("\nBest parameters (grid search):", best_config)


# ---------- (c) k-distance graph ----------
from sklearn.neighbors import NearestNeighbors

min_pts_opt = best_config[0]
neighbors = NearestNeighbors(n_neighbors=min_pts_opt)
neighbors_fit = neighbors.fit(X)
distances, indices = neighbors_fit.kneighbors(X)

distances = np.sort(distances[:, -1])
plt.plot(distances)
plt.title("K-Distance Graph")
plt.xlabel("Points sorted by distance")
plt.ylabel(f"Distance to {min_pts_opt}th nearest neighbor")
plt.show()

# (choose epsilon from the elbow visually — or use best_config[1] here)
eps_opt = best_config[1]
labels_final = dbscan(X, eps_opt, min_pts_opt)


# ---------- (d) Visualization ----------
plt.figure(figsize=(7, 5))
plt.scatter(X[:, 0], X[:, 1], c=labels_final, cmap='rainbow')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('DBSCAN Clustering (from scratch)')
plt.show()

# ---------- Result Summary ----------
n_clusters = len(set(labels_final)) - (1 if -1 in labels_final else 0)
n_noise = list(labels_final).count(-1)

print(f"\nFinal number of clusters: {n_clusters}")
print(f"Noise points: {n_noise}")
