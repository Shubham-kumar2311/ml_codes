import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# -------------------- Setup --------------------
DATA_FILE = "salary_data.csv"   # must contain columns: YearsExperience, Salary
RESULT_DIR = "results"
PLOTS_DIR = os.path.join(RESULT_DIR, "plots")
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

TRAIN_PCTS = list(range(10, 100, 10))  # 10%,20%,...,90%
RANDOM_STATE = 42

# -------------------- Load Data --------------------
df = pd.read_csv(DATA_FILE)
X = df[["YearsExperience"]].values
y = df["Salary"].values

# -------------------- Storage --------------------
metrics = []
predictions = []

# for combined hypothesis plot
x_line = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color="black", s=20, alpha=0.6, label="Data")

# -------------------- Loop over splits --------------------
for p in TRAIN_PCTS:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=p/100, random_state=RANDOM_STATE
    )

    # model: Linear regression with GD
    model = SGDRegressor(
        max_iter=10000, tol=1e-6, eta0=0.001, learning_rate="constant",
        penalty=None, random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)

    theta1 = model.coef_[0]
    theta0 = model.intercept_[0]

    # predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # metrics
    rss_train = mean_squared_error(y_train, y_train_pred)
    rss_test = mean_squared_error(y_test, y_test_pred)
    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)

    metrics.append([p, 100-p, theta0, theta1, rss_train, rss_test, r2_train, r2_test])

    for xi, yi, ypi in zip(X_test.flatten(), y_test, y_test_pred):
        predictions.append([p, 100-p, xi, yi, ypi])

    # add hypothesis line to combined plot
    plt.plot(x_line, model.predict(x_line), label=f"{p}% train")

# -------------------- Save Excel --------------------
metrics_df = pd.DataFrame(metrics, columns=[
    "Train %", "Test %", "theta0", "theta1", "Mean RSS (Train)", "Mean RSS (Test)",
    "R² (Train)", "R² (Test)"
])
metrics_df.to_excel(os.path.join(RESULT_DIR, "metrics.xlsx"), index=False)

pred_df = pd.DataFrame(predictions, columns=[
    "Train %", "Test %", "X (YearsExperience)", "Actual y (Salary)", "Predicted y"
])
pred_df.to_excel(os.path.join(RESULT_DIR, "predictions.xlsx"), index=False)

# -------------------- Plots --------------------
# combined hypotheses
plt.xlabel("YearsExperience")
plt.ylabel("Salary")
plt.title("Hypotheses for Different Training Splits (SGDRegressor)")
plt.legend(ncol=2, fontsize=8)
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "all_hypotheses.png"))
plt.close()

# RSS vs Training %
plt.plot(metrics_df["Train %"], metrics_df["Mean RSS (Train)"], label="Train RSS")
plt.plot(metrics_df["Train %"], metrics_df["Mean RSS (Test)"], label="Test RSS")
plt.xlabel("Training %")
plt.ylabel("Mean RSS")
plt.title("Training % vs Mean RSS")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "rss_vs_train_pct.png"))
plt.close()

# R² vs Training %
plt.plot(metrics_df["Train %"], metrics_df["R² (Train)"], label="Train R²")
plt.plot(metrics_df["Train %"], metrics_df["R² (Test)"], label="Test R²")
plt.xlabel("Training %")
plt.ylabel("R²")
plt.title("Training % vs R²")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "r2_vs_train_pct.png"))
plt.close()

print("✅ Done. Results in 'results/' and plots in 'results/plots/'")
