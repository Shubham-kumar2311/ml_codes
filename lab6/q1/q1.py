import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from itertools import product

np.random.seed(42)

# ------------------ Utility Functions ------------------ #

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def cutoff(y_pred_probs):
    return (y_pred_probs >= 0.5).astype(int)

def accuracy(y_true, y_pred_probs):
    return np.mean(cutoff(y_pred_probs) == y_true)

def log_loss(y_true, y_pred_probs, eps=1e-15):
    y_pred_probs = np.clip(y_pred_probs, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred_probs) + (1 - y_true) * np.log(1 - y_pred_probs))

def confusion_matrix(y_true, y_pred, labels):
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    label_to_index = {label: idx for idx, label in enumerate(labels)}
    for yt, yp in zip(y_true, y_pred):
        cm[label_to_index[yt], label_to_index[yp]] += 1
    return cm

def plot_loss(loss_history, fold_idx, cls, save_dir="plots"):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure()
    plt.plot(loss_history)
    plt.title(f'Fold {fold_idx} - {cls} Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(f'{save_dir}/loss_fold{fold_idx}_{cls}.png')
    plt.close()

def plot_confusion(cm, labels, fold_idx, save_dir="plots"):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(5,4))
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.title(f'Fold {fold_idx} Confusion Matrix')
    plt.xticks(np.arange(len(labels)), labels)
    plt.yticks(np.arange(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(j, i, str(cm[i,j]), ha='center', va='center', color='red')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.colorbar()
    plt.savefig(f'{save_dir}/cm_fold{fold_idx}.png')
    plt.close()

def predict_ova(X, all_theta):
    """X: m x n, all_theta: num_classes x n"""
    probs = sigmoid(X.dot(all_theta.T))
    return np.array([np.argmax(p) for p in probs])

# ------------------ Batch Gradient Descent ------------------ #
def logistic_regression_bgd(X, y, alpha=0.1, rho=1e-6, max_epochs=100):
    m, n = X.shape
    theta = np.zeros(n)
    loss_history = []
    prev_loss = float("inf")
    for epoch in range(max_epochs):
        y_pred = sigmoid(np.dot(X, theta))
        grad = (1/m) * X.T.dot(y_pred - y)
        theta -= alpha * grad
        loss = log_loss(y, y_pred)
        loss_history.append(loss)
        if abs(prev_loss - loss) < rho:
            break
        prev_loss = loss
    return theta, loss_history

# ------------------ K-fold Split ------------------ #
def kfold_split(X, y, k=5):
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    folds = np.array_split(indices, k)
    return folds

# ------------------ Main ------------------ #
# Read Iris dataset
df = pd.read_csv("Iris.csv")  # Assuming CSV has columns: Id, SepalLengthCm, ..., Species
X = df.iloc[:, 1:5].values  # 4 features
y = df['Species'].values
classes = np.unique(y)
num_classes = len(classes)

# Normalize features
X_min = X.min(axis=0)
X_max = X.max(axis=0)
X = (X - X_min) / (X_max - X_min)

# Hyperparameters
alphas = [0.0001, 0.1]
rhos = [0.001, 0.01]
epochs_list = [50, 100]

k = 5
folds = kfold_split(X, y, k)
results = []

for fold_idx in range(k):
    test_idx = folds[fold_idx]
    train_idx = np.hstack([folds[i] for i in range(k) if i != fold_idx])
    
    # Split train and test
    X_train_full, y_train_full = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    # 10% validation
    val_size = int(0.1 * len(X_train_full))
    X_val, y_val = X_train_full[:val_size], y_train_full[:val_size]
    X_train, y_train = X_train_full[val_size:], y_train_full[val_size:]
    
    best_val_loss = float("inf")
    best_params = None
    best_theta_all = None
    all_loss_histories = {}
    
    # Hyperparameter tuning
    for alpha, rho, epochs in product(alphas, rhos, epochs_list):
        theta_all_classes = []
        loss_histories = {}
        val_loss_total = 0
        for cls in classes:
            y_train_bin = (y_train == cls).astype(int)
            theta, loss_history = logistic_regression_bgd(X_train, y_train_bin, alpha, rho, epochs)
            theta_all_classes.append(theta)
            loss_histories[cls] = loss_history
            
            # Validation loss
            y_val_bin = (y_val == cls).astype(int)
            val_loss_total += log_loss(y_val_bin, sigmoid(X_val.dot(theta)))
        
        if val_loss_total < best_val_loss:
            best_val_loss = val_loss_total
            best_params = (alpha, rho, epochs)
            best_theta_all = np.array(theta_all_classes)
            all_loss_histories = loss_histories
    
    # Plot loss per class
    for cls, loss_hist in all_loss_histories.items():
        plot_loss(loss_hist, fold_idx + 1, cls)
    
    # Predictions
    y_train_pred_idx = predict_ova(X_train, best_theta_all)
    y_test_pred_idx = predict_ova(X_test, best_theta_all)
    y_train_pred = classes[y_train_pred_idx]
    y_test_pred = classes[y_test_pred_idx]
    
    # Confusion matrices
    cm_train = confusion_matrix(y_train, y_train_pred, classes)
    cm_test = confusion_matrix(y_test, y_test_pred, classes)
    plot_confusion(cm_test, classes, fold_idx + 1)
    
    # Class-wise metrics
    for cls_idx, cls in enumerate(classes):
        TP_train = cm_train[cls_idx, cls_idx]
        FP_train = cm_train[:, cls_idx].sum() - TP_train
        FN_train = cm_train[cls_idx, :].sum() - TP_train
        Precision_train = TP_train / (TP_train + FP_train) if (TP_train + FP_train) != 0 else 0
        Recall_train = TP_train / (TP_train + FN_train) if (TP_train + FN_train) != 0 else 0
        Acc_train = accuracy((y_train==cls).astype(int), (y_train_pred==cls).astype(int))
        
        TP_test = cm_test[cls_idx, cls_idx]
        FP_test = cm_test[:, cls_idx].sum() - TP_test
        FN_test = cm_test[cls_idx, :].sum() - TP_test
        Precision_test = TP_test / (TP_test + FP_test) if (TP_test + FP_test) != 0 else 0
        Recall_test = TP_test / (TP_test + FN_test) if (TP_test + FN_test) != 0 else 0
        Acc_test = accuracy((y_test==cls).astype(int), (y_test_pred==cls).astype(int))
        
        results.append([fold_idx+1, cls, Acc_train, Acc_test, Precision_test, Recall_test])

# Save results
os.makedirs("results", exist_ok=True)
df_results = pd.DataFrame(results, columns=["Fold","Class","Train Acc","Test Acc","Precision","Recall"])
df_results.to_csv("results/iris_logistic_ova_results.csv", index=False)
print("Training complete! Results saved in 'results/' folder and plots in 'plots/' folder.")
