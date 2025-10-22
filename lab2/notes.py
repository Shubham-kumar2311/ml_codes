def ols_fit(X, y):
    x_mean, y_mean = np.mean(X), np.mean(y)
    w1 = np.sum((X - x_mean) * (y - y_mean)) / np.sum((X - x_mean) ** 2) 
    w0 = y_mean - w1 * x_mean
    return w0, w1

def rss(y, y_pred):
    return np.mean((y - y_pred) ** 2)

def r2(y, y_pred):
    ss_res = np.sum((y - y_pred) ** 2)             
    ss_tot = np.sum((y - np.mean(y)) ** 2) 
    return 1 - (ss_res / ss_tot)

#rₓᵧ = [ Σ (Xᵢ − X̄)(Yᵢ − Ȳ) ] / [ √( Σ (Xᵢ − X̄)² ) × √( Σ (Yᵢ − Ȳ)² ) ]
x_mean = X_data.mean()
y_mean = Y_data.mean()

#[ Σ (Xᵢ − X̄)(Yᵢ − Ȳ) ]
numerator = 0
for idx in range(total_patterns):
    numerator += (X_data[idx] - x_mean) * (Y_data[idx] - y_mean)

# √( Σ (Xᵢ − X̄)² )
std_x = 0
for x in X_data:
    std_x += (x - x_mean) **2
std_x = std_x ** (1/2)

# √( Σ (Yᵢ − Ȳ)² )
std_y = 0
for y in Y_data:
    std_y += (y - y_mean) **2
std_y = std_y ** (1/2)

correlation = numerator / (std_x * std_y)

print("Correlation is:" , correlation)
exp_std = df['YearsExperience'].std()
print("Correlation by inbuilt is:" , salary_exp_cov / (salary_std * exp_std))

mean_crop = df['Y'].mean()
sample_df = df[df['Y'] > mean_crop]


def split_data(X, y, train_percent):
    N = len(X)
    idx = np.arange(N)
    np.random.shuffle(idx)

    train_size = int(N * train_percent / 100)
    train_idx = idx[:train_size]
    test_idx = idx[train_size:]

    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]



def gradient_descent(X, y, alpha=0.001, iterations=10000):
    m = len(X)
    theta0, theta1 = 0, 0
    for _ in range(iterations):
        y_pred = theta0 + theta1 * X
        error = y_pred - y
        d_theta0 = (1/m) * np.sum(error)
        d_theta1 = (1/m) * np.sum(error * X)
        theta0 -= alpha * d_theta0
        theta1 -= alpha * d_theta1
    return theta0, theta1
