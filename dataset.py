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

    # Compute labels using broadcasting instead of loops
    # x value = samples[:, :1]
    # y value = samples[:, 1:2]
    # row = np.sum(samples[:, :1] > grid, axis=1)
    # col = np.sum(samples[:, 1:2] > grid, axis=1)
    labels = (
            np.sum(samples[:, :1] > grid, axis=1) * int(np.sqrt(classes)) +
            np.sum(samples[:, 1:2] > grid, axis=1)
    )

    return samples, labels


def load_dataset(path):
    dataset = np.load(path)
    data, label = dataset.files
    return dataset[data], dataset[label]
