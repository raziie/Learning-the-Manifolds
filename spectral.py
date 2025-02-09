import numpy as np
from plot_utils import plot_2d_data
from geo import KNearestNeighbors, EpsNeighborhood, _compute_distance_matrix
from dataset import load_dataset


class Spectral:
    def __init__(self, n_components, *, adj_calculator=KNearestNeighbors(5)):
        self.n_components = n_components
        self._adj_calculator = adj_calculator
        self.t = 1

    def _compute_weights(self, X, distances):
        pairwise_distances = _compute_distance_matrix(X)
        self.t = np.median(pairwise_distances ** 2)
        weights = np.exp(-(pairwise_distances**2) / self.t)
        # weights = np.ones(len(distances))
        return weights * np.where(distances == np.inf, 0, 1)

    def _compute_embedding(self, W):
        # axis = 0 or 1 because W is summetric
        # print(np.allclose(W, W.T, atol=1e-5))
        D = np.diag(np.sum(W, axis=0))

        # # Ensures the eigenvalues are within [0, 2], making the problem well-conditioned.
        # # Compute D^(-1/2) manually
        # D_sqrt_inv = np.diag(1.0 / np.sqrt(np.diag(D) + 1e-8))  # Avoid division by zero
        # # Compute normalized Laplacian
        # L = np.eye(W.shape[0]) - D_sqrt_inv @ W @ D_sqrt_inv
        # # Solve the eigenvalue problem
        # eigenvalues, eigenvectors = np.linalg.eig(L)
        # eigenvectors = D_sqrt_inv @ eigenvectors  # Transform back

        L = D - W
        D_sqrt_inv = np.diag(1.0 / np.sqrt(np.diag(D) + 1e-8))  # Avoid division by zero
        L_norm = D_sqrt_inv @ L @ D_sqrt_inv

        eigenvalues, eigenvectors = np.linalg.eig(L_norm)
        eigenvectors = D_sqrt_inv @ eigenvectors  # Transform back


        # L = D - W
        # # Solve the generalized eigenvalue problem L*y = lambda*D*y
        # eigenvalues, eigenvectors = np.linalg.eig(np.linalg.inv(D) @ L)

        Y = eigenvectors[:, 1:self.n_components + 1]
        return Y

    def fit_transform(self, X):
        nearest_neighbors = self._adj_calculator(X)
        nearest_neighbors = np.maximum(nearest_neighbors, nearest_neighbors.T)
        W = self._compute_weights(X, nearest_neighbors)
        # W = np.maximum(W, nearest_neighbors.T)
        Y = self._compute_embedding(W)
        return Y


if __name__ == "__main__":
    # TODO: Load swiss roll dataset
    # Load swiss roll dataset
    path = "datasets/swissroll.npz"
    data, labels = load_dataset(path)

    # Apply spectral
    spectral = Spectral(n_components=2, adj_calculator=KNearestNeighbors(10))
    data_2d = spectral.fit_transform(data)

    from sklearn.manifold import SpectralEmbedding

    spectral = SpectralEmbedding(n_components=2, affinity='nearest_neighbors', n_neighbors=10, random_state=42)
    spectral_transformed = spectral.fit_transform(data)
    spectral_error = np.linalg.norm(data_2d - spectral_transformed)
    print(f"spectral Error: {spectral_error:.2f}")

    # Visualize the 2D projection
    plot_2d_data(data_2d, labels, "Spectral Projection of Swiss Roll")
