import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix, eye
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll

import numpy as np
from geo import KNearestNeighbors

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
        self._adj_calculator = adj_calculator
        
    def _compute_weights(self, X, distances):
        # TODO: Compute weights for each point using least squares        
        return None
    
    def _compute_embedding(self, W):
        # TODO: Compute M
        
        # TODO: Find eigen vectors
        
        # TODO: Return the vectors corresponding to the smallest non-zero values
        return None
    
    def fit_transform(self, X):
        """
        Fit the LLE model to the dataset and reduce its dimensionality.

        Parameters:
        - X: numpy array, the dataset (m x n).
        """
        # TODO: Find nearest neighbors
        
        # TODO: Compute reconstruction weights
        
        # TODO: Compute embedding
        return None


if __name__ == "__main__":
    # TODO: Load swiss roll dataset
    # TODO: Perform LLE
    # TODO: Visualize the results
    pass
