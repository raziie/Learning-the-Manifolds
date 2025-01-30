import numpy as np


def _compute_distance_matrix(X):
    # TODO: Compute pairwise Euclidean distance matrix for X
    squared_sum = np.sum(X**2, axis=1, keepdims=True)
    # d(xi, xj) = sqrt(sum((xi,k - xj,k)^2))
    # = sqrt(sum(xi,k^2) + sum(xj,k^2) - 2 * sum(xi,k xj,k))
    # = sqrt(||xi||^2 + ||xj||^2 - 2 * xi xj^T)
    distance_matrix = np.sqrt(squared_sum + squared_sum.T - 2 * X @ X.T)
    return distance_matrix


class KNearestNeighbors:
    """
    Compute the k-nearest neighbors for each point in the dataset.
    
    Attributes:
    - k: int, the number of nearest neighbors to find.
    """
    
    def __init__(self, k):
        self.k = k
    
    def __call__(self, X):
        """        
        Parameters:
        - X: numpy array, the dataset (m x n).
        
        Returns:
        - neighbors: numpy array, adjacency matrix (m x m).
        """
        # TODO: For each point, find the indices of the k smallest distances
        # Find distances
        distance_matrix = _compute_distance_matrix(X)
        # Sort distances
        # ?????????????????? it is not working correctly
        sorted_indices = np.argsort(distance_matrix, axis=1)

        # Initialize adjacency matrix with infinity
        m = distance_matrix.shape[0]
        neighbors = np.full((m, m), np.inf)
        # Find the place of k nearest neighbors
        condition = (sorted_indices <= self.k)
        # ???????????????????? is this needed?
        # & (sorted_indices > 0)
        # Set distances of k nearest ones
        neighbors[condition] = distance_matrix[condition]

        # or

        # # Set distances greater than epsilon to infinity
        # neighbors = np.where(sorted_indices <= self.k, distance_matrix, np.inf)

        return neighbors


class EpsNeighborhood:
    """
    Compute the epsilon-neighborhood for each point in the dataset.
    
    Attributes:
    - epsilon: float, the maximum distance to consider a point as a neighbor.
    """
    
    def __init__(self, eps):
        self.eps = eps
    
    def __call__(self, X):
        """
        Parameters:
        - X: numpy array, the dataset (m x n).
        
        Returns:
        - neighbors: numpy array, adjacency matrix (m x m).
        """
        # TODO: For each point, find the indices of points within the epsilon distance
        # Find distances
        distance_matrix = _compute_distance_matrix(X)

        # Initialize adjacency matrix with infinity
        m = distance_matrix.shape[0]
        neighbors = np.full((m, m), np.inf)
        # Find the place of neighbors with distance less than eps
        condition = (distance_matrix <= self.eps)
        # ???????????????????? is this needed?
        # & (distance_matrix > 0)
        # Set distances of nearest ones
        neighbors[condition] = distance_matrix[condition]

        # or

        # # Set distances greater than epsilon to infinity
        # neighbors = np.where(distance_matrix <= self.eps, distance_matrix, np.inf)

        return neighbors


if __name__ == "__main__":
    # data = np.random.randn(4, 6)
    data = np.random.randint(10, size=(4, 6))
    print(data)

    # print(_compute_distance_matrix(data))

    knn = KNearestNeighbors(2)
    print(knn(data))

    eps_neighborhood = EpsNeighborhood(10)
    print(eps_neighborhood(data))
