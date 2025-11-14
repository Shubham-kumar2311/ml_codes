import numpy as np

def initialize_som(m, n, dim):
    weights = np.random.rand(m, n, dim)
    grid = np.array([[i, j] for i in range(m) for j in range(n)]).reshape(m, n, 2)
    return weights, grid

def decay_radius(sigma, t, max_iter):
    return sigma * np.exp(-t / max_iter)

def decay_lr(lr, t, max_iter):
    return lr * np.exp(-t / max_iter)

def neighborhood(grid, bmu_idx, radius):
    d = np.sum((grid - bmu_idx) ** 2, axis=2)
    return np.exp(-d / (2 * radius ** 2))[..., np.newaxis]

def find_bmu(weights, x):
    dist = np.linalg.norm(weights - x, axis=2)
    return np.unravel_index(np.argmin(dist), weights.shape[:2])

def train_som(data, m=5, n=5, dim=None, lr=0.5, sigma=None, max_iter=1000):
    if dim is None:
        dim = data.shape[1]
    if sigma is None:
        sigma = max(m, n) / 2
    weights, grid = initialize_som(m, n, dim)
    for t in range(max_iter):
        x = data[np.random.randint(0, len(data))]
        bmu_idx = find_bmu(weights, x)
        r = decay_radius(sigma, t, max_iter)
        current_lr = decay_lr(lr, t, max_iter)
        h = neighborhood(grid, np.array(bmu_idx), r)
        weights += current_lr * h * (x - weights)
    return weights, grid

def map_data(weights, data):
    return np.array([find_bmu(weights, x) for x in data])

# Example usage
if __name__ == "__main__":
    from sklearn.datasets import load_iris
    X = load_iris().data
    X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

    weights, grid = train_som(X, m=5, n=5, dim=X.shape[1], lr=0.5, max_iter=1000)
    bmu_indices = map_data(weights, X)
