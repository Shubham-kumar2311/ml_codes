import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv("Synthetic dataset.csv")
print("Dataset loaded successfully!\n")
# print(df.head(), "\n")

# -----------------------------
# Column categories
# -----------------------------
numeric_cols = ["age", "income"]
categorical_cols = ["occupation", "region"]
ordinal_cols = ["education_level", "satisfaction"]

# -----------------------------
# Normalize numeric columns
# -----------------------------
for col in numeric_cols:
    df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

# -----------------------------
# Encode ordinal columns
# -----------------------------
education_map = {"High School": 1, "Bachelor": 2, "Master": 3, "PhD": 4}
satisfaction_map = {"Low": 1, "Medium": 2, "High": 3, "Very High": 4}
df["education_level"] = df["education_level"].map(education_map)
df["satisfaction"] = df["satisfaction"].map(satisfaction_map)

# -----------------------------
# Encode categorical columns
# -----------------------------
for col in categorical_cols:
    unique_vals = df[col].unique()
    mapping = {v: i for i, v in enumerate(unique_vals)}
    df[col] = df[col].map(mapping)

# -----------------------------
# Convert to numpy
# -----------------------------
data = df[numeric_cols + categorical_cols + ordinal_cols].values

# print(data)

num_idx = [0,1]
cat_idx = [2,3]
ord_idx = [4,5]

# -----------------------------
# Distance functions
# -----------------------------
def euclidean(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def manhattan(a, b):
    return np.sum(np.abs(a - b))

def hamming(a, b):
    return np.mean(a != b)

def mixed_distance(a, b):
    num_d = euclidean(a[num_idx], b[num_idx]) / np.sqrt(len(num_idx))
    cat_d = hamming(a[cat_idx], b[cat_idx])
    ord_d = manhattan(a[ord_idx], b[ord_idx]) / len(ord_idx)
    return num_d + cat_d + ord_d

# -----------------------------
# K-Means clustering
# -----------------------------
def kmeans(K, max_iter=100):
    n = len(data)
    centroids = data[np.random.choice(n, K, replace=False)]
    clusters = np.zeros(n, dtype=int)

    for _ in range(max_iter):
        new_clusters = np.zeros(n, dtype=int)

        # Assign points to nearest centroid
        for i in range(n):
            distances = [mixed_distance(data[i], c) for c in centroids]
            new_clusters[i] = np.argmin(distances)

        # Stop if clusters unchanged
        if np.array_equal(new_clusters, clusters):
            break
        clusters = new_clusters

        # Update centroids using medoid approach
        for k in range(K):
            members = data[clusters == k]
            if len(members) > 0:
                dist_matrix = np.array([[mixed_distance(a, b) for b in members] for a in members])
                centroids[k] = members[np.argmin(dist_matrix.sum(axis=1))]

    return clusters, centroids  # Return centroids as well

# -----------------------------
# Silhouette score
# -----------------------------
def silhouette_score(labels):
    n = len(data)
    sil = np.zeros(n)
    for i in range(n):
        same_cluster = [mixed_distance(data[i], data[j]) for j in range(n) if labels[j] == labels[i] and j != i]
        a = np.mean(same_cluster) if same_cluster else 0

        other_means = []
        for k in np.unique(labels):
            if k != labels[i]:
                other = [mixed_distance(data[i], data[j]) for j in range(n) if labels[j] == k]
                if other:
                    other_means.append(np.mean(other))
        b = min(other_means) if other_means else 0

        sil[i] = 0 if max(a, b) == 0 else (b - a) / max(a, b)
    return np.mean(sil)

# -----------------------------
# Run K-Means and evaluate
# -----------------------------
K_values = range(2, 8)
sil_scores = []

for K in K_values:
    print(f"Running K = {K}")
    labels, centroids = kmeans(K)
    # print(labels)
    score = silhouette_score(labels)
    sil_scores.append(score)
    print(f"Silhouette Score = {score:.4f}")

    # Visualize clusters using first 2 numeric features
    plt.figure()
    for k in range(K):
        cluster_points = data[labels == k]
        plt.scatter(cluster_points[:, num_idx[0]], cluster_points[:, num_idx[1]], label=f'Cluster {k+1}')
    plt.scatter(centroids[:, num_idx[0]], centroids[:, num_idx[1]], color='black', marker='x', s=150, label='Centroids')
    plt.xlabel(numeric_cols[0])
    plt.ylabel(numeric_cols[1])
    plt.title(f'K-Means Clustering (K={K})')
    plt.legend()
    plt.show()

# -----------------------------
# Plot silhouette scores
# -----------------------------
plt.plot(K_values, sil_scores, marker='o')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score vs K")
plt.grid(True)
plt.show()

best_K = K_values[np.argmax(sil_scores)]
print(f"\nOptimal K = {best_K} (Silhouette = {max(sil_scores):.4f})")
