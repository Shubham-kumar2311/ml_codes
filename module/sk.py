import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, r2_score, silhouette_score, adjusted_rand_score
)
from sklearn.datasets import load_iris, load_diabetes, make_blobs
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering

# ---------- Classification ----------
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

classifiers = {
    "LogisticRegression": LogisticRegression(max_iter=200),
    "RandomForest": RandomForestClassifier(),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC()
}

for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(f"\n{name} Metrics (Classification):")
    print("Accuracy:", round(accuracy_score(y_test, y_pred), 3))
    print("Precision:", round(precision_score(y_test, y_pred, average='weighted', zero_division=0),3))
    print("Recall:", round(recall_score(y_test, y_pred, average='weighted', zero_division=0),3))
    print("F1 Score:", round(f1_score(y_test, y_pred, average='weighted', zero_division=0),3))

# ---------- Regression ----------
X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

regressors = {
    "LinearRegression": LinearRegression(),
    "RandomForestRegressor": RandomForestRegressor(),
    "KNNRegressor": KNeighborsRegressor()
}

for name, reg in regressors.items():
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    print(f"\n{name} Metrics (Regression):")
    print("MSE:", round(mean_squared_error(y_test, y_pred),3))
    print("R²:", round(r2_score(y_test, y_pred),3))

# ---------- Clustering ----------
X, y_true = make_blobs(n_samples=150, n_features=4, centers=3, random_state=42)

clusterers = {
    "KMeans": KMeans(n_clusters=3, n_init=10, random_state=42),
    "DBSCAN": DBSCAN(eps=1.5, min_samples=5),
    "Agglomerative": AgglomerativeClustering(n_clusters=3)
}

for name, clusterer in clusterers.items():
    labels = clusterer.fit_predict(X)
    sil = silhouette_score(X, labels) if len(set(labels))>1 else np.nan
    ari = adjusted_rand_score(y_true, labels)
    print(f"\n{name} Metrics (Clustering):")
    print("Silhouette Score:", round(sil,3))
    print("Adjusted Rand Index:", round(ari,3))
