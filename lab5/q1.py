import numpy as np

def perceptron_train(X, y, lr=0.1, epochs=100):
    w = np.zeros(X.shape[1])
    for _ in range(epochs):
        for i in range(X.shape[0]):
            z = np.dot(X[i], w)
            y_pred = 1 if z >= 0 else 0
            w += lr * (y[i] - y_pred) * X[i]
    return w

def perceptron_predict(X, w):
    z = np.dot(X, w)
    return (z >= 0).astype(int)

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

X = np.hstack([np.ones((X.shape[0], 1)), X])

y = np.array([0, 0, 0, 1])

w = perceptron_train(X, y, lr=0.1, epochs=10)
print("Learned weights:", w)

preds = perceptron_predict(X, w)
print("Predictions:", preds)
