import numpy as np
from geo import KNearestNeighbors
from pca import PCA


def dijkstra(root, adjacency_matrix):
    n = adjacency_matrix.shape[0]
    # Track unvisited nodes
    unvisited = np.ones(n, dtype=bool)
    visited = np.zeros(n, dtype=bool)
    distances = adjacency_matrix[root].copy()

    while np.any(unvisited):
        # Select the nearest unvisited node
        min_dist_idx = np.argmin(np.where(visited, np.inf, distances))

        if distances[min_dist_idx] == 0:
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
        # TODO: Apply MDS (eigen-decomposition) to the geodesic distance matrix
        return None

    def fit_transform(self, X):
        """
        Fit the Isomap model to the dataset and reduce its dimensionality.

        Parameters:
        - X: numpy array, the dataset (m x n).
        """

        # TODO: Compute the distance matrix

        # TODO: Construct the adjacency graph

        # TODO: Perform dimensionality reduction on geodesic distances

        # TODO: Return transformed data


if __name__ == "__main__":
    # Define a small dataset
    # X_test = np.array([[0, 0], [1, 0], [2, 0], [3, 0]])
    X_test = np.array([
        [0, 0],  # Point 0
        [1, 1],  # Point 1
        [2, 2],  # Point 2
        [8, 8],  # Point 3 (far from others)
    ])

    # Create an Isomap instance with KNN adjacency calculation
    isomap = Isomap(n_components=2, adj_calculator=KNearestNeighbors(3))

    # Compute geodesic distances
    geodesic_distances = isomap._compute_geodesic_distances(X_test)

    # Print the computed geodesic distance matrix
    print("Geodesic Distance Matrix:")
    print(geodesic_distances)
