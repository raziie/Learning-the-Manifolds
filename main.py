import numpy as np
import matplotlib.pyplot as plt

from pca import PCA
from isomap import Isomap
from lle import LLE

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
    #     "Isomap(K=10)": Isomap(n_components=2, adj_calculator=KNearestNeighbors(10)),
    #     "Isomap(K=8)": Isomap(n_components=2, adj_calculator=KNearestNeighbors(8)),
    #     "Isomap(e=10)": Isomap(n_components=2, adj_calculator=EpsNeighborhood(10)),
    #     "LLE(K=10)": LLE(n_components=2, adj_calculator=KNearestNeighbors(10)),
    #     "LLE(K=8)": LLE(n_components=2, adj_calculator=KNearestNeighbors(8)),
    #     "LLE(e=10)": LLE(n_components=2, adj_calculator=EpsNeighborhood(10))
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
    #
    # # TODO: visualize and show the results
    # # Visualization
    # fig, axes = plt.subplots(2, int(len(algorithms)/2), figsize=(15, 5))
    # for ax, (name, (transformed_data, T_k)) in zip(axes, results.items()):
    #     scatter = ax.scatter(transformed_data[:, 0], transformed_data[:, 1], c=labels, cmap='Spectral', s=15)
    #     ax.set_title(f"{name} (T={T_k:.4f})")
    #     ax.set_xlabel("Component 1")
    #     ax.set_ylabel("Component 2")
    #
    # plt.colorbar(scatter, ax=axes, orientation='vertical', fraction=0.02)
    # plt.show()

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
        "Isomap(K=10)": Isomap(n_components=n_components, adj_calculator=KNearestNeighbors(20)),
        "Isomap(K=8)": Isomap(n_components=n_components, adj_calculator=KNearestNeighbors(18)),
        "Isomap(e=10)": Isomap(n_components=n_components, adj_calculator=EpsNeighborhood(20)),
        "LLE(K=10)": LLE(n_components=n_components, adj_calculator=KNearestNeighbors(20)),
        "LLE(K=8)": LLE(n_components=n_components, adj_calculator=KNearestNeighbors(18)),
        "LLE(e=10)": LLE(n_components=n_components, adj_calculator=EpsNeighborhood(20))
    }

    # Iterate over each algorithm and compute trustworthiness
    results = {}
    for name, algorithm in algorithms.items():
        transformed_data = algorithm.fit_transform(data)

        # Compute trustworthiness score
        T_k = trustworthiness(data, transformed_data, n_neighbors=10)
        results[name] = (transformed_data, T_k)
        print(f"{name} Trustworthiness: {T_k:.4f}")

    # Visualization of reduced data
    fig, axes = plt.subplots(2, int(len(algorithms) / 2), figsize=(15, 10))
    for ax, (name, (transformed_data, trustworthiness_score)) in zip(axes.flatten(), results.items()):
        # Plot images at the 2D coordinates
        for i in range(len(transformed_data)):
            image = data[i].reshape(32, 32)  # Reshape the flattened data into the 32x32 image
            ax.imshow(image,aspect='auto')  # Adjust as necessary for image scaling

        ax.set_title(f"{name} (T={trustworthiness_score:.4f})")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")

    plt.tight_layout()
    plt.show()
