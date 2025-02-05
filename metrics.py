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

    # Get the k-nearest neighbors in the original and embedded spaces
    orig_neighbors = np.argsort(D_orig, axis=1)[:, 1:n_neighbors + 1]
    emb_neighbors = np.argsort(D_emb, axis=1)[:, 1:n_neighbors + 1]

    # Compute ranks in the embedded space
    # Adding + 1 ensures ranks start from 1 instead of 0
    ranks = np.argsort(np.argsort(D_emb, axis=1), axis=1) + 1

    # Compute trustworthiness penalty
    # find misplaced neighbors
    misplaced_mask = ~np.isin(orig_neighbors, emb_neighbors)
    # compute how far they moved
    misplaced_ranks = ranks[np.arange(n)[:, None], orig_neighbors] * misplaced_mask
    penalty = np.sum(np.maximum(0, misplaced_ranks - n_neighbors))

    # Compute final trustworthiness score
    T = 1 - (2 / (n * n_neighbors * (2 * n - 3 * n_neighbors - 1))) * penalty

    from sklearn.manifold import trustworthiness
    print(f"sklearn:{trustworthiness(D, D_embedded, n_neighbors=5):.2f}")

    return T
