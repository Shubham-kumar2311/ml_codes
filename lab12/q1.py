import numpy as np


X = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]])
Y = np.array([[0],
              [1],
              [1],
              [0]])


X = np.hstack([np.ones((X.shape[0], 1)), X])  # shape (4, 3)


X_train, Y_train = X, Y
X_test, Y_test = X, Y


lr = 0.1
epochs = 2
np.random.seed(42)


W1 = np.random.uniform(-1, 1, (3, 2))   # input(3) → hidden(2)
W2 = np.random.uniform(-1, 1, (3, 1))   # hidden(2+1 bias) → output(1)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_deriv(z):
    return z * (1 - z)


for epoch in range(epochs):
    for i in range(len(X_train)):
        x = X_train[i].reshape(1, -1)
        y = Y_train[i].reshape(1, -1)
        
        # Forward
        hidden_in = np.dot(x, W1)
        hidden_out = sigmoid(hidden_in)
        hidden_out = np.hstack([np.ones((1,1)), hidden_out])  # add bias

        final_in = np.dot(hidden_out, W2)
        final_out = sigmoid(final_in)
        
        # Backward
        error = y - final_out
        d_final = error * sigmoid_deriv(final_out)

        d_hidden = np.dot(d_final, W2[1:].T) * sigmoid_deriv(hidden_out[:,1:])

        # Update
        W2 += lr * np.dot(hidden_out.T, d_final)
        W1 += lr * np.dot(x.T, d_hidden)
    
    print(f"Epoch {epoch+1} done")


def predict(X, W1, W2):
    hidden = sigmoid(np.dot(X, W1))
    hidden = np.hstack([np.ones((X.shape[0],1)), hidden])
    out = sigmoid(np.dot(hidden, W2))
    return out

y_pred_train = predict(X_train, W1, W2)
y_pred_test = predict(X_test, W1, W2)

# Threshold
y_pred_train_cls = (y_pred_train > 0.5).astype(int)
y_pred_test_cls = (y_pred_test > 0.5).astype(int)


def metrics(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    
    acc = (TP + TN) / len(y_true)
    prec = TP / (TP + FP) if (TP + FP) > 0 else 0
    rec = TP / (TP + FN) if (TP + FN) > 0 else 0
    return acc, prec, rec

a_train, p_train, r_train = metrics(Y_train, y_pred_train_cls)
a_test, p_test, r_test = metrics(Y_test, y_pred_test_cls)

print("\nAfter 2 Epochs:")
print(f"Train -> Accuracy: {a_train:.2f}, Precision: {p_train:.2f}, Recall: {r_train:.2f}")
print(f"Test  -> Accuracy: {a_test:.2f}, Precision: {p_test:.2f}, Recall: {r_test:.2f}")
