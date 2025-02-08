import numpy as np
import matplotlib.pyplot as plt

from pca import PCA
from isomap import Isomap
from lle import LLE
from spectral import Spectral

from geo import KNearestNeighbors, EpsNeighborhood
from dataset import load_dataset
from metrics import trustworthiness


if __name__ == "__main__":
    # # TODO: load the swiss roll dataset
    # # Load swiss roll dataset
    # path = "datasets/swissroll.npz"
    # data, labels = load_dataset(path)
    #
    # # TODO: try with different dimensionality reduction algorithms and different parameters
    # # Define dimensionality reduction methods
    # algorithms = {
    #     "PCA": PCA(n_components=2),
    #     "Isomap(K=20)": Isomap(n_components=2, adj_calculator=KNearestNeighbors(20)),
    #     "Isomap(K=18)": Isomap(n_components=2, adj_calculator=KNearestNeighbors(18)),
    #     "Isomap(e=20)": Isomap(n_components=2, adj_calculator=EpsNeighborhood(20)),
    #     "LLE(K=20)": LLE(n_components=2, adj_calculator=KNearestNeighbors(20)),
    #     "LLE(K=18)": LLE(n_components=2, adj_calculator=KNearestNeighbors(18)),
    #     "LLE(e=20)": LLE(n_components=2, adj_calculator=EpsNeighborhood(20)),
    #     "Spectral(K=20)": Spectral(n_components=2, adj_calculator=KNearestNeighbors(20)),
    #     "Spectral(K=10)": Spectral(n_components=2, adj_calculator=KNearestNeighbors(10)),
    #     "Spectral(e=20)": Spectral(n_components=2, adj_calculator=EpsNeighborhood(20))
    # }
    #
    # # TODO: calculate trustworthiness for each combination
    # # Iterate over each algorithm and compute trustworthiness
    # results = {}
    # for name, algorithm in algorithms.items():
    #     transformed_data = algorithm.fit_transform(data)
    #
    #     # Compute trustworthiness score
    #     T_k = trustworthiness(data, transformed_data, n_neighbors=5)
    #     results[name] = (transformed_data, T_k)
    #     print(f"{name} Trustworthiness: {T_k:.4f}")


    # Load ORL faces dataset
    path = "datasets/faces.npz"
    data, labels = load_dataset(path)

    n_components = 100

    # Apply PCA
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(data)
    # Reconstruct images using inverse_transform
    reconstructed_data = pca.inverse_transform(reduced_data)

    # only works with 2 components
    # Visualize Original vs Reconstructed Images
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    for i in range(5):
        axes[0, i].imshow(data[i].reshape(32, 32), cmap='gray')
        axes[0, i].set_title("Original")
        axes[0, i].axis("off")

        axes[1, i].imshow(reconstructed_data[i].reshape(32, 32), cmap='gray')
        axes[1, i].set_title("Reconstructed")
        axes[1, i].axis("off")

    plt.show()

    cumulative_evr = np.cumsum(pca.explained_variance_ratio_)[:n_components]

    # Plot cumulative explained variance ratio
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, n_components + 1), cumulative_evr, marker='o', linestyle='-')
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance Ratio")
    plt.title("Explained Variance Ratio for ORL Faces")
    plt.grid()
    plt.show()

    # Define dimensionality reduction methods
    algorithms = {
        "PCA": PCA(n_components=n_components),
        "Isomap(K=20)": Isomap(n_components=n_components, adj_calculator=KNearestNeighbors(20)),
        "Isomap(K=18)": Isomap(n_components=n_components, adj_calculator=KNearestNeighbors(18)),
        "Isomap(e=20)": Isomap(n_components=n_components, adj_calculator=EpsNeighborhood(20)),
        "LLE(K=20)": LLE(n_components=n_components, adj_calculator=KNearestNeighbors(20)),
        "LLE(K=18)": LLE(n_components=n_components, adj_calculator=KNearestNeighbors(18)),
        "LLE(e=20)": LLE(n_components=n_components, adj_calculator=EpsNeighborhood(20)),
        "Spectral(K=20)": Spectral(n_components=n_components, adj_calculator=KNearestNeighbors(20)),
        "Spectral(K=10)": Spectral(n_components=n_components, adj_calculator=KNearestNeighbors(10)),
        "Spectral(e=20)": Spectral(n_components=n_components, adj_calculator=EpsNeighborhood(20))
    }

    spectral = Spectral(n_components=2, adj_calculator=KNearestNeighbors(10))
    data_2d = spectral.fit_transform(data)


    # Iterate over each algorithm and compute trustworthiness
    results = {}
    for name, algorithm in algorithms.items():
        transformed_data = algorithm.fit_transform(data)

        # Compute trustworthiness score
        T_k = trustworthiness(data, transformed_data, n_neighbors=10)
        results[name] = (transformed_data, T_k)
        print(f"{name} Trustworthiness: {T_k:.4f}")
