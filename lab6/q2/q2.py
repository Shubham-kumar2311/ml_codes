import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt


data = pd.read_csv('wine-clustering.csv')
X = data.values    


def distance(a, b):
    diff = a - b
    return (np.sum(diff * diff)) ** 0.5

def mean(points):
    if len(points) == 0:
        return None
    return np.sum(points, axis=0) / len(points)


def Kmean(k, max_iters=100):
    indices = np.random.choice(len(X), k, replace=False)
    centroids = [X[i].copy() for i in indices]
    
    for _ in range(max_iters):
        clusters = [[] for _ in range(k)]
    
        for point in X:
            dists = [distance(point, c) for c in centroids]
            nearest = np.argmin(dists)
            clusters[nearest].append(point)
        
        new_centroids = []
        for cluster in clusters:
            m = mean(cluster)
            if m is None: 
                m = X[np.random.randint(0, len(X))]
            new_centroids.append(m)
        
        converged = True
        for i in range(k):
            if distance(centroids[i], new_centroids[i]) > 1e-6:
                converged = False
                break
        if converged:
            break
        
        centroids = new_centroids
    
    return centroids, clusters


Ks = [3, 5, 10, 15]


for k in Ks:
    centroids, clusters = Kmean(k)
    
    plt.figure()
    for i, cluster in enumerate(clusters):
        cluster = np.array(cluster)
        if len(cluster) == 0:
            continue
        plt.scatter(cluster[:, 0], cluster[:, 1], label=f'Cluster {i+1}')
    
    centroids = np.array(centroids)
    plt.scatter(centroids[:, 0], centroids[:, 1], color='black', marker='x', s=150, label='Centroids')
    
    plt.title(f'K-Means Clustering (k={k})')
    plt.xlabel('Feature 0')
    plt.ylabel('Feature 1')
    plt.legend()
    plt.show()
