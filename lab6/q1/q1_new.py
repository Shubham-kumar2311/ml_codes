import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from itertools import product
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, accuracy_score

np.random.seed(42)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def log_loss(y_true, y_pred, eps=1e-15):
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def plot_loss(loss_hist, fold, cls, save_dir="plots"):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure()
    plt.plot(loss_hist)
    plt.title(f"Fold {fold} - {cls} Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(f"{save_dir}/loss_fold{fold}_{cls}.png")
    plt.close()

def train_logistic_bgd(X, y, alpha=0.1, rho=1e-6, max_epochs=100):
    theta = np.zeros(X.shape[1])
    loss_history = []
    prev_loss = float("inf")
    for _ in range(max_epochs):
        y_pred = sigmoid(X.dot(theta))
        grad = X.T.dot(y_pred - y) / len(y)
        theta -= alpha * grad
        loss = log_loss(y, y_pred)
        loss_history.append(loss)
        if abs(prev_loss - loss) < rho:
            break
        prev_loss = loss
    return theta, loss_history

def predict_ova(X, all_theta):
    probs = sigmoid(X.dot(all_theta.T))
    return np.argmax(probs, axis=1)

df = pd.read_csv("Iris.csv")
X = df.iloc[:, 1:5].values
y = df['Species'].values
classes = np.unique(y)

X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

alphas = [0.01, 0.1]
rhos = [0.001, 0.01]
epochs_list = [50, 100]

kf = KFold(n_splits=5, shuffle=True, random_state=42)
results = []

for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X), 1):
    X_train_full, X_test = X[train_idx], X[test_idx]
    y_train_full, y_test = y[train_idx], y[test_idx]
    
    val_size = int(0.1 * len(X_train_full))
    X_val, y_val = X_train_full[:val_size], y_train_full[:val_size]
    X_train, y_train = X_train_full[val_size:], y_train_full[val_size:]
    
    best_val_loss = float("inf")
    best_theta_all = None
    
    for alpha, rho, epochs in product(alphas, rhos, epochs_list):
        theta_all = []
        val_loss_total = 0
        loss_histories = {}
        for cls in classes:
            y_train_bin = (y_train == cls).astype(int)
            theta, loss_hist = train_logistic_bgd(X_train, y_train_bin, alpha, rho, epochs)
            theta_all.append(theta)
            loss_histories[cls] = loss_hist
            y_val_bin = (y_val == cls).astype(int)
            val_loss_total += log_loss(y_val_bin, sigmoid(X_val.dot(theta)))
        
        if val_loss_total < best_val_loss:
            best_val_loss = val_loss_total
            best_theta_all = np.array(theta_all)
            best_loss_histories = loss_histories
    
    for cls, loss_hist in best_loss_histories.items():
        plot_loss(loss_hist, fold_idx, cls)
    
    y_train_pred_idx = predict_ova(X_train, best_theta_all)
    y_test_pred_idx = predict_ova(X_test, best_theta_all)
    y_train_pred = classes[y_train_pred_idx]
    y_test_pred = classes[y_test_pred_idx]
    
    cm_test = confusion_matrix(y_test, y_test_pred, labels=classes)
    os.makedirs("plots", exist_ok=True)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=classes)
    disp.plot(cmap="Blues")
    plt.title(f"Fold {fold_idx} Confusion Matrix")
    plt.savefig(f"plots/cm_fold{fold_idx}.png")
    plt.close()
    
    for cls in classes:
        y_train_bin = (y_train == cls).astype(int)
        y_train_pred_bin = (y_train_pred == cls).astype(int)
        y_test_bin = (y_test == cls).astype(int)
        y_test_pred_bin = (y_test_pred == cls).astype(int)
        
        Acc_train = accuracy_score(y_train_bin, y_train_pred_bin)
        Precision_test = precision_score(y_test_bin, y_test_pred_bin, zero_division=0)
        Recall_test = recall_score(y_test_bin, y_test_pred_bin, zero_division=0)
        Acc_test = accuracy_score(y_test_bin, y_test_pred_bin)
        
        results.append([fold_idx, cls, Acc_train, Acc_test, Precision_test, Recall_test])

os.makedirs("results", exist_ok=True)
df_results = pd.DataFrame(results, columns=["Fold","Class","Train Acc","Test Acc","Precision","Recall"])
df_results.to_csv("results/iris_logistic_ova_results.csv", index=False)
print("Training complete! Results saved in 'results/' folder and plots in 'plots/' folder.")
