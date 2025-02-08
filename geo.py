import numpy as np


def _compute_distance_matrix(X):
    # TODO: Compute pairwise Euclidean distance matrix for X
    squared_sum = np.sum(X**2, axis=1, keepdims=True)
    # d(xi, xj) = sqrt(sum((xi,k - xj,k)^2))
    # = sqrt(sum(xi,k^2) + sum(xj,k^2) - 2 * sum(xi,k xj,k))
    # = sqrt(||xi||^2 + ||xj||^2 - 2 * xi xj^T)
    distance_matrix = squared_sum + squared_sum.T - 2 * X @ X.T
    # distance_matrix = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=-1)

    # Clip negative values caused by floating-point errors
    distance_matrix = np.maximum(distance_matrix, 0)

    return np.sqrt(distance_matrix)


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

        # Mask out the diagonal by setting it to infinity
        np.fill_diagonal(distance_matrix, np.inf)



        # # to consider all points in the same distance and not just the first one
        # # Sort distances for each row (each point)
        # sorted_indices = np.argsort(distance_matrix, axis=1)  # Indices of sorted distances
        # sorted_distances = np.sort(distance_matrix, axis=1)  # Sorted distances
        #
        # # Get the distance of the k-th nearest neighbor
        # kth_distance = sorted_distances[:, self.k - 1:self.k]  # Shape (m,1)
        #
        # # Create a mask where distances <= k-th nearest distance
        # mask = distance_matrix <= kth_distance  # Shape (m,m), True where edge should exist
        #
        # # Construct adjacency matrix with distances, set non-edges to np.inf
        # neighbors = np.where(mask, distance_matrix, np.inf)
        # np.fill_diagonal(neighbors, 0)



        # ??????????????????? what if two neighbors are in equal distances?
        # Sort distances and get the nearest k ones
        k_nearest_neighbors = np.argsort(distance_matrix, axis=1)[:, :self.k]

        # Initialize adjacency matrix with infinity
        m = distance_matrix.shape[0]
        neighbors = np.full((m, m), np.inf)
        np.fill_diagonal(neighbors, 0)
        # Set distances of k nearest ones
        # Use advanced indexing to replace the distances for the k nearest neighbors
        neighbors[np.arange(m)[:, None], k_nearest_neighbors] = distance_matrix[
            np.arange(m)[:, None], k_nearest_neighbors]

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

        # Set distances greater than epsilon to infinity
        neighbors = np.where(distance_matrix <= self.eps, distance_matrix, np.inf)

        return neighbors


if __name__ == "__main__":
    # # data = np.random.randn(4, 6)
    # data = np.random.randint(10, size=(4, 6))
    # # data = np.array([
    # #     [0, 0],  # Point 0
    # #     [1, 1],  # Point 1
    # #     [2, 2],  # Point 2
    # #     [8, 8],  # Point 3 (far from others)
    # # ])
    # # data = np.array([
    # #     [0, 0],
    # #     [1, 0],
    # #     [2, 0],
    # #     [3, 0]
    # # ])
    #
    # print(data)
    #
    # # print(_compute_distance_matrix(data))
    #
    # knn = KNearestNeighbors(2)
    # print(knn(data))
    #
    # eps_neighborhood = EpsNeighborhood(5)
    # print(eps_neighborhood(data))

    # not correct output
    X = np.array([[0, 0], [1, 0], [2, 0]])
    knn = KNearestNeighbors(k=1)
    adjacency_matrix = knn(X)
    print("Adjacency Matrix:\n", adjacency_matrix)
