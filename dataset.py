import numpy as np


def generate_plane(d=2, dim=3, classes=2, n_samples=500, noise_std=0.1):
    # TODO: generate a noisy d-dimensional plane within
    # a dim-dimensional space partitioned into classes
    if d > dim:
        raise ValueError("The hyperplane dimension 'd' must be less than or equal to the ambient dimension 'dim'.")

    # random basis vectors V
    coefficients = np.random.randn(dim, d)
    # generate sample points Vx = y
    samples = np.random.randn(n_samples, d) @ coefficients.T
    # add gaussian noise
    samples += np.random.normal(scale=noise_std, size=samples.shape)

    grid = np.linspace(-1, 1, int(np.sqrt(classes)) + 1)[1:-1]
    labels = np.zeros(n_samples, dtype=int)

    for i, point in enumerate(samples):
        # Use the first two dimensions of the points to assign labels
        x, y = point[:2]
        # Determine the grid cell (row, col) and calculate a unique label
        row = sum((x > g) for g in grid)
        col = sum((y > g) for g in grid)
        labels[i] = row * int(np.sqrt(classes)) + col

    return samples, labels


def load_dataset(path):
    dataset = np.load(path)
    data, label = dataset.files
    return dataset[data], dataset[label]
