import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from math import sqrt

np.random.seed(42)

os.makedirs("results", exist_ok=True)
os.makedirs("plots", exist_ok=True)

data = pd.read_csv("Mall_Customers.csv")
X = data[['Annual Income (k$)', 'Spending Score (1-100)']].values

def euclidean_distance(a, b):
    return sqrt(np.sum((a - b) ** 2))

def fuzzy_c_means(X, n_clusters=5, m=2.0, max_iter=100, error=1e-5):
    n, d = X.shape
    U = np.random.rand(n, n_clusters)
    U = U / np.sum(U, axis=1, keepdims=True)

    for _ in range(max_iter):
        U_m = U ** m
        centers = (U_m.T @ X) / np.sum(U_m.T, axis=1, keepdims=True)
        dist = np.zeros((n, n_clusters))
        for i in range(n):
            for j in range(n_clusters):
                dist[i, j] = euclidean_distance(X[i], centers[j]) + 1e-10

        new_U = np.zeros(U.shape)
        for i in range(n):
            for j in range(n_clusters):
                denom = np.sum((dist[i, j] / dist[i, :]) ** (2 / (m - 1)))
                new_U[i, j] = 1 / denom

        if np.sum((new_U - U) ** 2) ** 0.5 < error:
            break
        U = new_U
    return centers, U

centers, U = fuzzy_c_means(X, n_clusters=5)
labels = np.argmax(U, axis=1)


results_df = pd.DataFrame(X, columns=['Annual Income (k$)', 'Spending Score (1-100)'])
for i in range(U.shape[1]):
    results_df[f'Membership_Cluster_{i+1}'] = U[:, i]
results_df['Cluster_Label'] = labels
results_df.to_csv("results/fuzzy_cmeans_results.csv", index=False)

centers_df = pd.DataFrame(centers, columns=['Annual Income (k$)', 'Spending Score (1-100)'])
centers_df.to_csv("results/cluster_centers.csv", index=False)


plt.figure(figsize=(7, 6))
for i in range(5):
    plt.scatter(X[labels == i, 0], X[labels == i, 1])
plt.scatter(centers[:, 0], centers[:, 1], c='black', marker='X', s=200)
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Fuzzy C-Means Clustering')
plt.savefig("plots/fuzzy_cmeans_clusters.png", dpi=300)
plt.show()


fpc = np.sum(U ** 2) / len(X)
print(f"FPC: {fpc:.4f}")
print("Centers:\n", centers)
print("\n Results saved to 'results/' and plot saved to 'plots/'.")
