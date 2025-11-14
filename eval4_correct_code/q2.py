import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

np.random.seed(0)
os.makedirs("result2", exist_ok=True)

X = np.random.randn(300, 2)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

def split_data(X, y, train=0.7, val=0.1, test=0.2):
    n = len(y)
    idx = np.random.permutation(n)
    X, y = X[idx], y[idx]
    t1, t2 = int(train*n), int((train+val)*n)
    return X[:t1], y[:t1], X[t1:t2], y[t1:t2], X[t2:], y[t2:]

X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y)

def perceptron_train(X, y, w_init, epochs):
    X = np.c_[np.ones(X.shape[0]), X]
    w = w_init.copy()
    for _ in range(epochs):
        errors = 0
        for xi, target in zip(X, y):
            y_pred = 1 if np.dot(w, xi) > 0 else 0
            if y_pred != target:
                errors += 1
                if y_pred == 0 and target == 1:
                    w += xi
                elif y_pred == 1 and target == 0:
                    w -= xi
        if errors == 0:
            break
    return w

def perceptron_predict(X, w):
    X = np.c_[np.ones(X.shape[0]), X]
    return (np.dot(X, w) > 0).astype(int)

def precision(y_true, y_pred):
    TP = np.sum((y_true==1)&(y_pred==1))
    FP = np.sum((y_true==0)&(y_pred==1))
    return TP/(TP+FP) if (TP+FP)>0 else 0

def recall(y_true, y_pred):
    TP = np.sum((y_true==1)&(y_pred==1))
    FN = np.sum((y_true==1)&(y_pred==0))
    return TP/(TP+FN) if (TP+FN)>0 else 0

def accuracy(y_true, y_pred):
    return np.mean(y_true==y_pred)

epochs_list = [50,100,150]
results = []

for run in range(3):
    w_init = np.random.uniform(-0.5,0.5,3)
    for ep in epochs_list:
        w = perceptron_train(X_train, y_train, w_init, ep)
        y_pred = perceptron_predict(X_test, w)
        results.append({
            "Init_Run": run+1,
            "Epochs": ep,
            "Precision": precision(y_test, y_pred),
            "Recall": recall(y_test, y_pred),
            "Accuracy": accuracy(y_test, y_pred)
        })

df = pd.DataFrame(results)
df.to_csv("result2/perceptron_results.csv", index=False)

for metric in ["Precision","Recall","Accuracy"]:
    plt.figure()
    for run in df["Init_Run"].unique():
        subset = df[df["Init_Run"]==run]
        plt.plot(subset["Epochs"], subset[metric], marker='o', label=f"Init {run}")
    plt.title(metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend()
    plt.savefig(f"result2/{metric}.png")
    plt.close()
