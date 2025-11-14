import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression, LinearRegression

# ---------- Classification ----------
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy:", round(accuracy_score(y_test, y_pred), 3))
print("Precision:", round(precision_score(y_test, y_pred, average='weighted', zero_division=0),3))
print("Recall:", round(recall_score(y_test, y_pred, average='weighted', zero_division=0),3))
print("F1 Score:", round(f1_score(y_test, y_pred, average='weighted', zero_division=0),3))

