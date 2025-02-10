import numpy as np
from geo import KNearestNeighbors, EpsNeighborhood, _compute_distance_matrix
from dataset import load_dataset
from plot_utils import plot_2d_data


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
            C += reg * np.eye(k) * (np.trace(C) if np.trace(C) > 0 else 1)

            # Solve for the weights using least squares (CW = 1)
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
        eigenvalues, eigenvectors = np.linalg.eig(M)
        # TODO: Return the vectors corresponding to the smallest non-zero values
        # Sort eigenvalues and eigenvectors in ascending order
        sorted_indices = np.argsort(eigenvalues.real)
        eigenvalues = eigenvalues.real[sorted_indices]
        eigenvectors = eigenvectors.real[:, sorted_indices]

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
        # distance_matrix = _compute_distance_matrix(X)
        # nearest_neighbors = np.where(self._adj_calculator(X) == 1, distance_matrix, 0)
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

    # TODO: Perform LLE
    # Apply LLE
    lle = LLE(n_components=2, adj_calculator=KNearestNeighbors(10))
    data_2d = lle.fit_transform(data)

    # TODO: Visualize the results
    # Visualize the 2D projection
    plot_2d_data(data_2d, labels, "LLE Projection of Swiss Roll")

    # from sklearn.manifold import LocallyLinearEmbedding
    # # Compare LLE
    # lle = LocallyLinearEmbedding(n_components=2, n_neighbors=20)
    # sklearn_transformed = lle.fit_transform(data)
    # lle_error = np.linalg.norm(data_2d - sklearn_transformed)
    # print(f"lle Error: {lle_error:.2f}")
