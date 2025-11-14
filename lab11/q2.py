import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

os.makedirs("results2", exist_ok=True)
os.makedirs("plots2", exist_ok=True)

data = pd.read_csv("Iris.csv")
X = data[["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]].values
X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
Y = data["Species"].values
classes = np.unique(Y)
Y_onehot = np.zeros((len(Y), len(classes)))
for i, c in enumerate(classes):
    Y_onehot[Y == c, i] = 1

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict(X, W):
    return sigmoid(np.dot(X, W))

def train_slp(X, Y, lr=0.1, epochs=200):
    W = np.random.uniform(-0.5, 0.5, (X.shape[1], Y.shape[1]))
    for _ in range(epochs):
        y_pred = predict(X, W)
        W += lr * np.dot(X.T, (Y - y_pred))
    return W

def metrics(y_true, y_pred):
    y_true_cls = np.argmax(y_true, axis=1)
    y_pred_cls = np.argmax(y_pred, axis=1)
    acc = np.mean(y_true_cls == y_pred_cls)
    prec_list, rec_list = [], []
    for c in np.unique(y_true_cls):
        TP = np.sum((y_pred_cls == c) & (y_true_cls == c))
        FP = np.sum((y_pred_cls == c) & (y_true_cls != c))
        FN = np.sum((y_pred_cls != c) & (y_true_cls == c))
        prec = TP / (TP + FP) if (TP + FP) > 0 else 0
        rec = TP / (TP + FN) if (TP + FN) > 0 else 0
        prec_list.append(prec)
        rec_list.append(rec)
    return np.mean(prec_list), np.mean(rec_list), acc

kf = KFold(n_splits=5, shuffle=True, random_state=42)
results = []

for train_idx, test_idx in kf.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    Y_train, Y_test = Y_onehot[train_idx], Y_onehot[test_idx]
    X_train = np.hstack([np.ones((X_train.shape[0],1)), X_train])
    X_test = np.hstack([np.ones((X_test.shape[0],1)), X_test])
    W = train_slp(X_train, Y_train)
    y_pred_train = predict(X_train, W)
    y_pred_test = predict(X_test, W)
    p_train, r_train, a_train = metrics(Y_train, y_pred_train)
    p_test, r_test, a_test = metrics(Y_test, y_pred_test)
    results.append([p_train, r_train, a_train, p_test, r_test, a_test])

results = np.array(results)
train_prec, train_rec, train_acc = results[:,0].mean(), results[:,1].mean(), results[:,2].mean()
test_prec, test_rec, test_acc = results[:,3].mean(), results[:,4].mean(), results[:,5].mean()

df = pd.DataFrame({
    "Set": ["Train", "Test"],
    "Precision": [train_prec, test_prec],
    "Recall": [train_rec, test_rec],
    "Accuracy": [train_acc, test_acc]
})
df.to_csv("results2/results.csv", index=False)

plt.figure(figsize=(6,4))
plt.bar(["Train","Test"], [train_acc, test_acc], color=["steelblue","orange"])
plt.title("Accuracy (5-Fold Mean)")
plt.ylabel("Accuracy")
plt.tight_layout()
plt.savefig("plots2/accuracy.png")
plt.close()

plt.figure(figsize=(6,4))
x = np.arange(2)
width = 0.35
plt.bar(x - width/2, [train_prec, test_prec], width, label="Precision")
plt.bar(x + width/2, [train_rec, test_rec], width, label="Recall")
plt.xticks(x, ["Train","Test"])
plt.ylabel("Score")
plt.title("Precision vs Recall (5-Fold Mean)")
plt.legend()
plt.tight_layout()
plt.savefig("plots2/prec_rec.png")
plt.close()
    