import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

data = pd.read_csv("Iris.csv")
X = data[["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]].values
X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
Y = data['Species'].values

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def precision(y_true, y_pred):
    labels = np.unique(y_true)
    precisions = {}
    for label in labels:
        TP = np.sum((y_true == label) & (y_pred == label))
        FP = np.sum((y_true != label) & (y_pred == label))
        precisions[label] = TP / (TP + FP) if (TP + FP) > 0 else 0
    return precisions

def recall(y_true, y_pred):
    labels = np.unique(y_true)
    recalls = {}
    for label in labels:
        TP = np.sum((y_true == label) & (y_pred == label))
        FN = np.sum((y_true == label) & (y_pred != label))
        recalls[label] = TP / (TP + FN) if (TP + FN) > 0 else 0
    return recalls

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def knn_predict(X_train, y_train, X_test, k):
    preds = []
    for x in X_test:
        d = [euclidean_distance(x, x_train) for x_train in X_train]
        idx = np.argsort(d)[:k]
        lbls = y_train[idx]
        vals, cnts = np.unique(lbls, return_counts=True)
        preds.append(vals[np.argmax(cnts)])
    return np.array(preds)

def weighted_knn_predict(X_train, y_train, X_test, k, power):
    preds = []
    for x in X_test:
        d = np.array([euclidean_distance(x, x_train) for x_train in X_train])
        idx = np.argsort(d)[:k]
        lbls = y_train[idx]
        d = d[idx]
        weights = 1 / (d**power + 1e-6)
        unique_labels = np.unique(lbls)
        weight_sum = {label: np.sum(weights[lbls == label]) for label in unique_labels}
        preds.append(max(weight_sum, key=weight_sum.get))
    return np.array(preds)

def split_data(X, y, train_ratio=0.8):
    n = len(y)
    idx = np.random.permutation(n)
    X, y = X[idx], y[idx]
    split = int(train_ratio * n)
    return X[:split], y[:split], X[split:], y[split:]

X_train, y_train, X_test, y_test = split_data(X, Y, 0.8)
K = np.random.randint(1, int(np.sqrt(len(y_train)))+1, 5)
os.makedirs("result1", exist_ok=True)

methods = {"Normal": None, "1/d": 1, "1/d^2": 2}
results = []

for name, power in methods.items():
    for k in K:
        if power is None:
            y_pred = knn_predict(X_train, y_train, X_test, k)
        else:
            y_pred = weighted_knn_predict(X_train, y_train, X_test, k, power)
        prec = precision(y_test, y_pred)
        rec = recall(y_test, y_pred)
        acc = accuracy(y_test, y_pred)
        results.append({"Method": name, "K": k, "Accuracy": acc, **{f"P_{l}": prec[l] for l in prec}, **{f"R_{l}": rec[l] for l in rec}})

df = pd.DataFrame(results)
df.to_csv("result1/final_results.csv", index=False)

for metric in ["Accuracy"] + [c for c in df.columns if c.startswith("P_")] + [c for c in df.columns if c.startswith("R_")]:
    plt.figure()
    for name in methods:
        subset = df[df["Method"] == name]
        plt.plot(subset["K"], subset[metric], marker='o', label=name)
    plt.title(metric)
    plt.xlabel("K")
    plt.ylabel(metric)
    plt.legend()
    plt.savefig(f"result1/{metric}.png")
    plt.close()
