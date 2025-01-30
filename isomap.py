import numpy as np
from geo import KNearestNeighbors
from pca import PCA


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
        return None

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
