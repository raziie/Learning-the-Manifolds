import numpy as np
from geo import KNearestNeighbors
from pca import PCA
from dataset import load_dataset
from matplotlib import pyplot as plt


def dijkstra(root, adjacency_matrix):
    n = adjacency_matrix.shape[0]
    # Track unvisited nodes
    unvisited = np.ones(n, dtype=bool)
    visited = np.zeros(n, dtype=bool)
    distances = adjacency_matrix[root].copy()

    while np.any(unvisited):
        # Select the nearest unvisited node
        unvisited_distances = np.where(visited, np.inf, distances)
        min_dist_idx = np.argmin(unvisited_distances)

        if unvisited_distances[min_dist_idx] == np.inf:
            # Remaining nodes are unreachable
            break
        visited[min_dist_idx] = True
        unvisited[min_dist_idx] = False

        # Vectorized distance update
        new_dist = distances[min_dist_idx] + adjacency_matrix[min_dist_idx]
        distances = np.minimum(distances, new_dist)

    return distances


class Isomap:
    """
    Isomap for dimensionality reduction by preserving geodesic distances.
    """

    def __init__(self, n_components, *, adj_calculator=KNearestNeighbors(5), decomposer=None):
        """
        Initialize Isomap with the number of components and neighbors.

        Parameters:
        - n_components: int, the number of dimensions to retain in the reduced space.
        - adj_calculator: function, given a dataset, returns the adjacency matrix.
        """
        # TODO: initialize required instance variables.
        self._adj_calculator = adj_calculator
        self._decomposer = decomposer or PCA(n_components=n_components)

    def _compute_geodesic_distances(self, X):
        # TODO: Use a shortest-path algorithm to compute the geodesic distances
        adjacency_matrix = self._adj_calculator(X)
        n = adjacency_matrix.shape[0]
        # geodesic_graph initialized with infinity
        geodesic_graph = np.full((n, n), np.inf)
        # Distance to self is 0
        np.fill_diagonal(geodesic_graph, 0)

        for i in range(n):
            geodesic_graph[i] = dijkstra(i, adjacency_matrix)
        return geodesic_graph

    def _decompose(self, geodesic_distances):
        # ????????????????? what does this comment mean?
        # TODO: Apply MDS (eigen-decomposition) to the geodesic distance matrix
        n = geodesic_distances.shape[0]
        I = np.eye(n)
        J = np.ones((n, n))
        C = np.subtract(I, (1/n)*J)
        # C = I - (1 / n) * J
        D_squared = np.square(geodesic_distances)
        # D2 = D ** 2
        B = (-1/2) * (C @ D_squared @ C)

        # apply PCA
        transformed_data = self._decomposer.fit_transform(B)
        return transformed_data

    def fit_transform(self, X):
        """
        Fit the Isomap model to the dataset and reduce its dimensionality.

        Parameters:
        - X: numpy array, the dataset (m x n).
        """

        # again don't get the comments meaning
        # TODO: Compute the distance matrix
        # TODO: Construct the adjacency graph
        geodesic_graph = self._compute_geodesic_distances(X)

        # TODO: Perform dimensionality reduction on geodesic distances
        transformed_data = self._decompose(geodesic_graph)

        # TODO: Return transformed data
        return transformed_data


if __name__ == "__main__":
    # # Define a small dataset
    # X_test = np.array([[0, 0], [1, 0], [2, 0], [3, 0]])
    # # X_test = np.array([
    # #     [0, 0],  # Point 0
    # #     [1, 1],  # Point 1
    # #     [2, 2],  # Point 2
    # #     [8, 8],  # Point 3 (far from others)
    # # ])
    #
    # # Create an Isomap instance with KNN adjacency calculation
    # isomap = Isomap(n_components=2, adj_calculator=KNearestNeighbors(3))
    #
    # # Compute geodesic distances
    # geodesic_distances = isomap._compute_geodesic_distances(X_test)
    #
    # # Print the computed geodesic distance matrix
    # print("Geodesic Distance Matrix:")
    # print(geodesic_distances)
    #
    # isomap._decompose(geodesic_distances)


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

    # Apply the Isomap algorithm to project the data into a 2-dimensional space
    isomap = Isomap(n_components=2, adj_calculator=KNearestNeighbors(3))
    data_2d = isomap.fit_transform(data)

    from sklearn.manifold import Isomap as SklearnIsomap
    # Compare Isomap
    sklearn_isomap = SklearnIsomap(n_components=2)
    sklearn_transformed = sklearn_isomap.fit_transform(data)
    print("Scikit-Learn Isomap:\n", sklearn_transformed)
    isomap_error = np.linalg.norm(data_2d - sklearn_transformed)
    print(f"isomap Error: {isomap_error:.2f}")

    # Visualize the 2D projection
    plt.figure(figsize=(8, 6))
    plt.scatter(data_2d[:, 0], data_2d[:, 1], c=labels, cmap='Spectral', s=15)
    plt.colorbar(label="Labels")
    plt.title("PCA Projection of Swiss Roll")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.show()


    # # not correct output
    # X = np.array([[0, 0], [1, 0], [2, 0]])
    # isomap = Isomap(n_components=2, adj_calculator=KNearestNeighbors(1))
    # geodesic_distances = isomap._compute_geodesic_distances(X)
    # print("Geodesic Distance Matrix:\n", geodesic_distances)
    #
    # X = np.array([[0, 0], [1, 0], [2, 0]])
    # isomap = Isomap(n_components=2, adj_calculator=KNearestNeighbors(1))
    # transformed_data = isomap.fit_transform(X)
    # print("Transformed Data:\n", transformed_data)