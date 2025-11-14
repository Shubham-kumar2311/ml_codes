


def demo_sklearn_metrics():
    import numpy as np
    import pandas as pd
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        mean_squared_error, r2_score, silhouette_score, adjusted_rand_score
    )
    from sklearn.cluster import KMeans


  

    # ---------- Classification Example ----------
    y_true_cls = [0, 1, 1, 0]
    y_pred_cls = [0, 1, 0, 0]


    # Per-class precision
    prec_per_class = precision_score(y_true_cls, y_pred_cls, average=None)
    print("Precision per class:", prec_per_class)


    acc = accuracy_score(y_true_cls, y_pred_cls)
    prec = precision_score(y_true_cls, y_pred_cls, average='macro', zero_division=0)
    rec = recall_score(y_true_cls, y_pred_cls, average='macro', zero_division=0)
    f1 = f1_score(y_true_cls, y_pred_cls, average='macro', zero_division=0)
    print("Classification Metrics:")
    print(f"Accuracy: {acc}, Precision: {prec}, Recall: {rec}, F1: {f1}\n")

    df_cls = pd.DataFrame({
        "Accuracy": [acc],
        "Precision": [prec],
        "Recall": [rec],
        "F1": [f1]
    })
    df_cls.to_csv("results/classification_metrics.csv", index=False)

    # ---------- Regression Example ----------
    y_true_reg = np.array([2.5, 0.0, 2.1, 7.8])
    y_pred_reg = np.array([3.0, -0.1, 2.0, 7.8])
    mse = mean_squared_error(y_true_reg, y_pred_reg)
    r2 = r2_score(y_true_reg, y_pred_reg)
    print("Regression Metrics:")
    print(f"MSE: {mse}, R2: {r2}\n")

    df_reg = pd.DataFrame({"MSE": [mse], "R2": [r2]})
    df_reg.to_csv("results/regression_metrics.csv", index=False)

    # ---------- Clustering Example ----------
    X_clust = np.array([[0,0],[0,1],[1,0],[5,5],[5,6],[6,5]])
    kmeans = KMeans(n_clusters=2, n_init=10, random_state=42).fit(X_clust)
    labels = kmeans.labels_
    sil_score = silhouette_score(X_clust, labels)
    ari = adjusted_rand_score([0,0,0,1,1,1], labels)  # true cluster labels
    print("Clustering Metrics:")
    print(f"Silhouette Score: {sil_score}, Adjusted Rand Index: {ari}\n")

    df_clust = pd.DataFrame({"Silhouette": [sil_score], "AdjustedRand": [ari]})
    df_clust.to_csv("results/clustering_metrics.csv", index=False)

# Run the demo
import os
os.makedirs("results", exist_ok=True)
demo_sklearn_metrics()


# --- 1.3 sklearn.model_selection (train_test_split, cross_val_score) ---
def demo_sklearn_model_selection():
    import numpy as np
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.linear_model import LogisticRegression

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    scores = cross_val_score(LogisticRegression(max_iter=1000), X, y, cv=3)
    print(scores.mean())


# --- 1.4 sklearn.preprocessing (StandardScaler, MinMax, LabelEncoder, OneHot, normalize) ---
def demo_sklearn_preprocessing():
    import numpy as np
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder, normalize

    X = np.array([[1.0], [2.0], [3.0]])
    print("StandardScaler:", StandardScaler().fit_transform(X))

    print("MinMaxScaler:", MinMaxScaler().fit_transform(X))

    le = LabelEncoder()
    print("LabelEncoder:", le.fit_transform(["cat", "dog", "cat"]))

    enc = OneHotEncoder(sparse_output=False)
    print("OneHot:", enc.fit_transform([["red"], ["blue"], ["red"]]))

    print("normalize:", normalize([[3, 4]]))




# --- 2.1 Matplotlib (pyplot, axes, figure) ---
def demo_matplotlib():
    import matplotlib.pyplot as plt
    x = [0, 1, 2]
    y = [0, 1, 0]
    plt.plot(x, y)
    plt.show()


# --- 2.2 Seaborn basic plots ---
def demo_seaborn():
    import seaborn as sns
    import pandas as pd
    import matplotlib.pyplot as plt
    df = pd.DataFrame({"x":[1,2,3,4],"y":[3,4,2,5],"g":["A","A","B","B"]})
    sns.scatterplot(data=df, x="x", y="y", hue="g")
    plt.show()



# =========================
# 3) NUMPY
# =========================

# --- 3.1 numpy core ---
def demo_numpy_core():
    import numpy as np
    a = np.array([[1,2],[3,4]])
    print(a.T)


# --- 3.2 numpy.random ---
def demo_numpy_random():
    import numpy as np
    print(np.random.rand(2,2))
    print(np.random.randint(0,10,3))


# --- 3.3 numpy.ma ---
def demo_numpy_ma():
    import numpy as np
    mask_arr = np.ma.masked_array([1,-1,3], mask=[0,1,0])
    print(mask_arr.filled(0))



# =========================
# 4) PANDAS
# =========================

# --- 4.1 pandas.core (Series, DataFrame) ---
def demo_pandas_core():
    import pandas as pd
    df = pd.DataFrame({"A":[1,2], "B":[3,4]})
    print(df)


# --- 4.2 pandas.io (read_csv, to_csv, read_json) ---
def demo_pandas_io():
    import pandas as pd
    df = pd.DataFrame({"A":[1,2]})
    df.to_csv("file.csv", index=False)
    print(pd.read_csv("file.csv"))


# --- 4.3 pandas.plotting (scatter_matrix) ---
def demo_pandas_plotting():
    import pandas as pd
    from pandas.plotting import scatter_matrix
    import matplotlib.pyplot as plt
    df = pd.DataFrame({"A":[1,2,3],"B":[4,5,6]})
    scatter_matrix(df)
    plt.show()


# --- 4.4 pandas.arrays (extension dtypes) ---
def demo_pandas_arrays():
    import pandas as pd
    s = pd.Series([1, None, 3], dtype="Int64")
    print(s)



# =========================
# 5) SCIPY
# =========================

# --- 5.1 scipy.cluster (hierarchy, kmeans) ---
def demo_scipy_cluster():
    import numpy as np
    from scipy.cluster.vq import kmeans2
    X = np.array([[0,0],[1,1],[5,5]])
    print(kmeans2(X, k=2))


# --- 5.2 scipy.constants ---
def demo_scipy_constants():
    from scipy import constants
    print(constants.pi)


# --- 5.3 scipy.sparse ---
def demo_scipy_sparse():
    import numpy as np
    from scipy.sparse import csr_matrix
    print(csr_matrix([[0,1],[1,0]]))


# --- 5.4 scipy.spatial ---
def demo_scipy_spatial():
    import numpy as np
    from scipy.spatial import distance
    X = np.array([[0,0],[1,1]])
    print(distance.cdist(X, X))


# --- 5.5 scipy.special ---
def demo_scipy_special():
    from scipy import special
    print(special.gamma(5))


# --- 5.6 scipy.stats ---
def demo_scipy_stats():
    from scipy import stats
    print(stats.ttest_ind([1,2,3],[3,4,5]))



# =========================
# 6) OPTUNA
# =========================
def demo_optuna():
    import optuna
    from sklearn.datasets import load_iris
    from sklearn.svm import SVC
    from sklearn.model_selection import cross_val_score

    X, y = load_iris(return_X_y=True)

    def objective(trial):
        C = trial.suggest_float("C", 0.1, 10)
        model = SVC(C=C)
        return cross_val_score(model, X, y, cv=3).mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=5)
    print(study.best_params)



# =============================================================
# Run anything here by uncommenting:
# =============================================================

if __name__ == "__main__":
    # demo_sklearn_covariance()
    # demo_sklearn_metrics()
    # demo_sklearn_model_selection()
    # demo_sklearn_preprocessing()
    # demo_matplotlib()
    # demo_seaborn()
    # demo_numpy_core()
    # demo_numpy_random()
    # demo_numpy_ma()
    # demo_pandas_core()
    # demo_pandas_io()
    # demo_pandas_plotting()
    # demo_pandas_arrays()
    # demo_scipy_cluster()
    # demo_scipy_constants()
    # demo_scipy_sparse()
    # demo_scipy_spatial()
    # demo_scipy_special()
    # demo_scipy_stats()
    # demo_optuna()
    pass
