import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv("diabetes.csv")  # Replace with your dataset
X = data.iloc[:, :-1].values
y_raw = data.iloc[:, -1].values
classes = np.unique(y_raw)
num_classes = len(classes)

Y = np.zeros((len(y_raw), num_classes))
for i, c in enumerate(classes):
    Y[y_raw == c, i] = 1

scaler = MinMaxScaler()
X = scaler.fit_transform(X)
X = np.hstack([np.ones((X.shape[0],1)), X])

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_deriv(z):
    return z * (1 - z)

def train_MLP(X, Y, lr=0.1, epochs=50, n_hidden=5):
    np.random.seed(42)
    n_inputs = X.shape[1]
    n_output = Y.shape[1]
    W1 = np.random.uniform(-1,1,(n_inputs, n_hidden))
    W2 = np.random.uniform(-1,1,(n_hidden+1, n_output))
    loss_history = {c: [] for c in range(n_output)}

    for epoch in range(epochs):
        for i in range(len(X)):
            x = X[i].reshape(1,-1)
            y = Y[i].reshape(1,-1)

            h_in = x.dot(W1)
            h_out = sigmoid(h_in)
            h_out = np.hstack([np.ones((1,1)), h_out])

            o_in = h_out.dot(W2)
            o_out = sigmoid(o_in)

            error = y - o_out
            d_o = error * sigmoid_deriv(o_out)
            d_h = d_o.dot(W2[1:].T) * sigmoid_deriv(h_out[:,1:])

            W2 += lr * h_out.T.dot(d_o)
            W1 += lr * x.T.dot(d_h)

            for c in range(n_output):
                loss = - (y[0,c]*np.log(o_out[0,c]+1e-15) + (1-y[0,c])*np.log(1-o_out[0,c]+1e-15))
                loss_history[c].append(loss)
    return W1, W2, loss_history

def predict(X, W1, W2):
    h = sigmoid(X.dot(W1))
    h = np.hstack([np.ones((X.shape[0],1)), h])
    o = sigmoid(h.dot(W2))
    return np.argmax(o, axis=1)

def metrics(y_true, y_pred):
    acc = np.mean(y_true == y_pred)
    return acc

def plot_losses(loss_history, fold):
    os.makedirs("plots", exist_ok=True)
    plt.figure()
    for c, losses in loss_history.items():
        epoch_loss = [np.mean(losses[i:i+len(losses)//50]) for i in range(0, len(losses), len(losses)//50)]
        plt.plot(epoch_loss, label=f"Class {c}")
    plt.title(f"Fold {fold} - Average Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"plots/loss_fold{fold}.png")
    plt.close()

def kfold_MLP(X, Y, epochs=50, n_hidden=5, k=5):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    results_all = []
    os.makedirs("results", exist_ok=True)

    for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]

        W1, W2, loss_history = train_MLP(X_train, Y_train, lr=0.1, epochs=epochs, n_hidden=n_hidden)
        plot_losses(loss_history, fold)

        y_train_pred = predict(X_train, W1, W2)
        y_test_pred = predict(X_test, W1, W2)
        y_train_true = np.argmax(Y_train, axis=1)
        y_test_true = np.argmax(Y_test, axis=1)

        for c_idx, c in enumerate(classes):
            TP_train = np.sum((y_train_pred==c_idx)&(y_train_true==c_idx))
            FP_train = np.sum((y_train_pred==c_idx)&(y_train_true!=c_idx))
            FN_train = np.sum((y_train_pred!=c_idx)&(y_train_true==c_idx))
            prec_train = TP_train/(TP_train+FP_train) if TP_train+FP_train>0 else 0
            rec_train = TP_train/(TP_train+FN_train) if TP_train+FN_train>0 else 0
            acc_train = np.mean(y_train_pred==y_train_true)

            TP_test = np.sum((y_test_pred==c_idx)&(y_test_true==c_idx))
            FP_test = np.sum((y_test_pred==c_idx)&(y_test_true!=c_idx))
            FN_test = np.sum((y_test_pred!=c_idx)&(y_test_true==c_idx))
            prec_test = TP_test/(TP_test+FP_test) if TP_test+FP_test>0 else 0
            rec_test = TP_test/(TP_test+FN_test) if TP_test+FN_test>0 else 0
            acc_test = np.mean(y_test_pred==y_test_true)

            results_all.append([fold, c, acc_train, acc_test, prec_train, rec_train, prec_test, rec_test])

    df_results = pd.DataFrame(results_all, columns=["Fold","Class","Train Acc","Test Acc","Train Prec","Train Rec","Test Prec","Test Rec"])
    df_results.to_csv("results/mlp_multiclass_results.csv", index=False)
    print("Training complete! Results saved in 'results/' and loss plots in 'plots/' folder.")

kfold_MLP(X, Y, epochs=100, n_hidden=5)
