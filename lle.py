import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix, eye
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll
from geo import KNearestNeighbors
from dataset import load_dataset


class LLE:
    """
    Locally Linear Embedding for nonlinear dimensionality reduction.
    """

    def __init__(self, n_components, *, adj_calculator=KNearestNeighbors(5)):
        """
        Initialize LLE with the number of components and neighbors.

        Parameters:
        - n_components: int, the number of dimensions to retain in the reduced space.
        - adj_calculator: function, given a dataset, returns the adjacency matrix.
        """
        # TODO: initialize required instance variables.
        self.n_components = n_components
        self._adj_calculator = adj_calculator

    def _compute_weights(self, X, distances, reg=1e-3):
        # TODO: Compute weights for each point using least squares
        m = X.shape[0]
        W = np.zeros((m, m))

        for i in range(m):
            # Get the k-nearest neighbors of point i
            condition = (distances[i] != np.inf) & (distances[i] != 0)
            Xi = X[condition] - X[i]

            # Compute Covariance matrix C = X @ X.T with regularization
            C = Xi @ Xi.T  # (k, k)
            k = C.shape[0]
            # Regularize to avoid singularities
            C += reg * np.eye(k) * np.trace(C)

            # Solve for the weights using least squares (GW = 1)
            w = np.linalg.solve(C, np.ones(k))  # (k,)

            # Normalize weights to sum to 1
            w = w / np.sum(w)

            # Store the weights in the weight matrix
            W[i, condition] = w

        return W

    def _compute_embedding(self, W):
        # TODO: Compute M
        m = W.shape[0]
        # Construct the cost matrix M = (I - W)^T (I - W)
        I = np.eye(m)
        M = (I - W).T @ (I - W)
        # TODO: Find eigen vectors
        # Perform eigenvalue decomposition on M
        eigenvalues, eigenvectors = np.linalg.eigh(M)
        # TODO: Return the vectors corresponding to the smallest non-zero values
        # Sort eigenvalues and eigenvectors in ascending order
        sorted_indices = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        # Discard the smallest eigenvalue and its corresponding eigenvector
        Y = eigenvectors[:, 1:self.n_components + 1]  # (m, n_components)
        return Y

    def fit_transform(self, X):
        """
        Fit the LLE model to the dataset and reduce its dimensionality.

        Parameters:
        - X: numpy array, the dataset (m x n).
        """
        # TODO: Find nearest neighbors
        nearest_neighbors = self._adj_calculator(X)
        # TODO: Compute reconstruction weights
        W = self._compute_weights(X, nearest_neighbors)
        # TODO: Compute embedding
        Y = self._compute_embedding(W)
        return Y


if __name__ == "__main__":
    # TODO: Load swiss roll dataset
    # Load swiss roll dataset
    path = "datasets/swissroll.npz"
    data, labels = load_dataset(path)

    # Visualize the Swiss Roll in 3D
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=labels, cmap='Spectral', s=15)
    legend1 = ax.legend(*scatter.legend_elements(), title="Labels")
    ax.add_artist(legend1)
    ax.set_title("Swiss Roll Dataset")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()

    # TODO: Perform LLE
    # Apply LLE
    lle = LLE(n_components=2, adj_calculator=KNearestNeighbors(20))
    data_2d = lle.fit_transform(data)

    from sklearn.manifold import LocallyLinearEmbedding
    # Compare LLE
    lle = LocallyLinearEmbedding(n_components=2, n_neighbors=20)
    sklearn_transformed = lle.fit_transform(data)
    lle_error = np.linalg.norm(data_2d - sklearn_transformed)
    print(f"lle Error: {lle_error:.2f}")

    # TODO: Visualize the results
    # Visualize the 2D projection
    plt.figure(figsize=(8, 6))
    plt.scatter(data_2d[:, 0], data_2d[:, 1], c=labels, cmap='Spectral', s=15)
    plt.colorbar(label="Labels")
    plt.title("LLE Projection of Swiss Roll")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.show()
