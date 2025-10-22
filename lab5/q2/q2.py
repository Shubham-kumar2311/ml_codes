import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from math import sqrt

np.random.seed(42)
os.makedirs("results", exist_ok=True)
os.makedirs("plots", exist_ok=True)

data = pd.read_csv("diabetes.csv")
X = data.values[:, :-1]
X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

y = data.values[:, -1]

def euclidean_distance(a, b):
    return sqrt(np.sum((a - b) ** 2))

def knn_predict(X_train, y_train, X_test, k=5):
    preds = []
    for x in X_test:
        d = [euclidean_distance(x, x_train) for x_train in X_train]
        idx = np.argsort(d)[:k]
        lbls = y_train[idx]
        vals, cnts = np.unique(lbls, return_counts=True)
        preds.append(vals[np.argmax(cnts)])
    return np.array(preds)

n = len(X)
split = int(0.8 * n)
idx = np.random.permutation(n)
train, test = idx[:split], idx[split:]
X_train, X_test = X[train], X[test]
y_train, y_test = y[train], y[test]

k = 5
y_pred = knn_predict(X_train, y_train, X_test, k)
acc = np.mean(y_pred == y_test)
print(f"Accuracy: {acc*100:.2f}%")

pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}).to_csv("results/knn_results.csv", index=False)

if X.shape[1] == 2:
    plt.figure(figsize=(7,6))
    plt.scatter(X_train[:,0], X_train[:,1], c=y_train, alpha=0.6)
    plt.scatter(X_test[:,0], X_test[:,1], c=y_pred, marker='X', s=100)
    plt.savefig("plots/knn_classification.png", dpi=300)
    plt.show()

print("Results saved to 'results/' and plot saved to 'plots/'.")
