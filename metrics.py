import numpy as np
from geo import _compute_distance_matrix


def trustworthiness(D, D_embedded, *, n_neighbors=5):
    """
    Computes the trustworthiness score to evaluate how well the local structure
    is preserved after dimensionality reduction.

    Parameters:
    - D: numpy array, the distance matrix in the original high-dimensional space.
    - D_embedded: numpy array, the distance matrix in the lower-dimensional space.
    - n_neighbors: int, the number of nearest neighbors to consider.

    Returns:
    - float: Trustworthiness score in the range [0, 1], where 1 indicates perfect preservation.
    """
    # TODO: Implement the trustworthiness calculation based on the formula
    n = D.shape[0]

    # Compute pairwise distance matrices
    D_orig = _compute_distance_matrix(D)
    D_emb = _compute_distance_matrix(D_embedded)

    # Set diagonal to np.inf to exclude self-neighbors
    np.fill_diagonal(D_orig, np.inf)
    # Get the ranking of neighbors in the original space
    orig_neighbors = np.argsort(D_orig, axis=1)
    emb_neighbors = np.argsort(D_emb, axis=1)[:, 1:n_neighbors + 1]

    # Build an inverted index: ranks of all points in original space
    inverted_index = np.zeros((n, n), dtype=int)
    ordered_indices = np.arange(n + 1)
    inverted_index[ordered_indices[:-1, np.newaxis], orig_neighbors] = ordered_indices[1:]
    # Compute ranks of embedded neighbors in the original space
    ranks = inverted_index[np.arange(n)[:, None], emb_neighbors]

    penalty = np.sum(np.maximum(0, ranks - n_neighbors))
    # Compute final trustworthiness score
    T = 1 - (2 / (n * n_neighbors * (2 * n - 3 * n_neighbors - 1))) * penalty

    # from sklearn.manifold import trustworthiness
    # out_sklearn = trustworthiness(D, D_embedded, n_neighbors=n_neighbors)
    # print(f"sklearn:{out_sklearn:.2f}")

    return T
