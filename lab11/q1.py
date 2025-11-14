import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

data = pd.read_csv("Iris.csv")

X = data[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]].values
X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
Y, species_map = pd.factorize(data["Species"])

X = np.hstack([np.ones((X.shape[0], 1)), X])

n = len(X)
n_train = int(0.7 * n)
n_val = int(0.1 * n)
idx = np.random.permutation(n)
X_train, y_train = X[idx[:n_train]], Y[idx[:n_train]]
X_val, y_val = X[idx[n_train:n_train + n_val]], Y[idx[n_train:n_train + n_val]]
X_test, y_test = X[idx[n_train + n_val:]], Y[idx[n_train + n_val:]]

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict(X, w):
    return sigmoid(np.dot(X, w))

def train_sigmoid(X, y, lr, epochs):
    w = np.zeros(X.shape[1])
    history = []
    for i in range(epochs):
        y_pred = predict(X, w)
        grad = np.dot(X.T, (y - y_pred)) / len(X)
        w += lr * grad
        loss = -np.mean(y * np.log(y_pred + 1e-9) + (1 - y) * np.log(1 - y_pred + 1e-9))
        history.append(loss)
    return w, history

def one_vs_all_train(X, y, lr, epochs):
    weights, histories = [], []
    for c in np.unique(y):
        y_bin = np.where(y == c, 1, 0)
        w, hist = train_sigmoid(X, y_bin, lr, epochs)
        weights.append(w)
        histories.append(hist)
    return np.array(weights), histories

def one_vs_all_predict(X, weights):
    scores = np.dot(X, weights.T)
    probs = sigmoid(scores)
    return np.argmax(probs, axis=1)

def precision_recall_accuracy(y_true, y_pred):
    classes = np.unique(y_true)
    prec, rec = {}, {}
    for c in classes:
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        prec[c] = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec[c] = tp / (tp + fn) if (tp + fn) > 0 else 0
    acc = np.mean(y_true == y_pred)
    return prec, rec, acc

os.makedirs("plots1", exist_ok=True)
os.makedirs("results1", exist_ok=True)

lrs = [0.01, 0.05, 0.1]
epochs_list = [100, 200, 300]
best_acc = 0
best_params = {}

for lr in lrs:
    for ep in epochs_list:
        w, histories = one_vs_all_train(X_train, y_train, lr, ep)
        y_val_pred = one_vs_all_predict(X_val, w)
        _, _, acc = precision_recall_accuracy(y_val, y_val_pred)
        if acc > best_acc:
            best_acc = acc
            best_params = {"lr": lr, "epochs": ep, "weights": w, "histories": histories}

w_best = best_params["weights"]
y_train_pred = one_vs_all_predict(X_train, w_best)
y_test_pred = one_vs_all_predict(X_test, w_best)

prec_train, rec_train, acc_train = precision_recall_accuracy(y_train, y_train_pred)
prec_test, rec_test, acc_test = precision_recall_accuracy(y_test, y_test_pred)

results_df = pd.DataFrame({
    "Dataset": ["Train", "Test"],
    "Accuracy": [acc_train, acc_test],
    **{f"Precision_{i}": [prec_train.get(i, 0), prec_test.get(i, 0)] for i in np.unique(Y)},
    **{f"Recall_{i}": [rec_train.get(i, 0), rec_test.get(i, 0)] for i in np.unique(Y)}
})
results_df.to_csv("results1/results.csv", index=False)

plt.figure(figsize=(8, 5))
for i, hist in enumerate(best_params["histories"]):
    plt.plot(hist, label=f"Class {i}")
plt.title("Loss vs Epochs (One-vs-All)")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("plots1/loss_curve.png")
plt.close()

cm = confusion_matrix(y_test, y_test_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=species_map)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix (Test Set)")
plt.tight_layout()
plt.savefig("plots1/confusion_matrix.png")
plt.close()
