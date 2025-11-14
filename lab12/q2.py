import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler


data = pd.read_csv("diabetes.csv")

X = data[["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
          "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]].values

Y = data[["Outcome"]].values

scaler = MinMaxScaler()
X = scaler.fit_transform(X)


X = np.hstack([np.ones((X.shape[0], 1)), X])


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_deriv(z):
    return z * (1 - z)



def train_MLP(X, Y, lr=0.1, epochs=50):
    np.random.seed(42)
    n_inputs = X.shape[1]
    n_hidden = 2
    n_output = 1

    W1 = np.random.uniform(-1, 1, (n_inputs, n_hidden))
    W2 = np.random.uniform(-1, 1, (n_hidden + 1, n_output))

    for epoch in range(epochs):
        for i in range(len(X)):
            x = X[i].reshape(1, -1)
            y = Y[i].reshape(1, -1)

            # Forward pass
            hidden_in = np.dot(x, W1)
            hidden_out = sigmoid(hidden_in)
            hidden_out = np.hstack([np.ones((1,1)), hidden_out])  # add bias

            final_in = np.dot(hidden_out, W2)
            final_out = sigmoid(final_in)

            # Backpropagation
            error = y - final_out
            d_final = error * sigmoid_deriv(final_out)
            d_hidden = np.dot(d_final, W2[1:].T) * sigmoid_deriv(hidden_out[:,1:])

            # Weight updates
            W2 += lr * np.dot(hidden_out.T, d_final)
            W1 += lr * np.dot(x.T, d_hidden)

    return W1, W2



def predict(X, W1, W2):
    hidden = sigmoid(np.dot(X, W1))
    hidden = np.hstack([np.ones((X.shape[0],1)), hidden])
    out = sigmoid(np.dot(hidden, W2))
    return (out > 0.5).astype(int)


def metrics(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    acc = (TP + TN) / len(y_true)
    prec_0 = TN / (TN + FN) if (TN + FN) > 0 else 0
    rec_0 = TN / (TN + FP) if (TN + FP) > 0 else 0
    prec_1 = TP / (TP + FP) if (TP + FP) > 0 else 0
    rec_1 = TP / (TP + FN) if (TP + FN) > 0 else 0
    return acc, prec_0, rec_0, prec_1, rec_1


def kfold_MLP(X, Y, epochs):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    accs, p0s, r0s, p1s, r1s = [], [], [], [], []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]

        W1, W2 = train_MLP(X_train, Y_train, lr=0.1, epochs=epochs)
        Y_pred = predict(X_test, W1, W2)

        acc, p0, r0, p1, r1 = metrics(Y_test, Y_pred)
        accs.append(acc)
        p0s.append(p0); r0s.append(r0)
        p1s.append(p1); r1s.append(r1)

        print(f"Fold {fold}: Acc={acc:.3f} | Class0(P,R)=({p0:.3f},{r0:.3f}) | Class1(P,R)=({p1:.3f},{r1:.3f})")

    print("\n=== Average across 5 folds ===")
    print(f"Class 0 → Precision={np.mean(p0s):.3f}, Recall={np.mean(r0s):.3f}")
    print(f"Class 1 → Precision={np.mean(p1s):.3f}, Recall={np.mean(r1s):.3f}")
    print(f"Overall Accuracy={np.mean(accs):.3f}\n")


print("Results for 50 Epochs:")
kfold_MLP(X, Y, epochs=50)

print("Results for 100 Epochs:")
kfold_MLP(X, Y, epochs=100)
