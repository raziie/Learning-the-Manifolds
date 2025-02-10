from joblib import dump
import matplotlib.pyplot as plt
import numpy as np

from pca import PCA
from isomap import Isomap
from lle import LLE
from spectral import Spectral

from geo import KNearestNeighbors, EpsNeighborhood
from dataset import load_dataset
from metrics import trustworthiness
from visualizer import Visualizer
from plot_utils import plot_images, plot_statistics


def apply_algorithms(input_data, input_parameters, file_name, n_components=2):
    # TODO: try with different dimensionality reduction algorithms and different parameters
    # Define dimensionality reduction methods
    # more k better but to some extent
    algorithms = {
        # more n_component better because bigger and more eigenvectors
        "PCA": PCA(n_components=n_components),
        f"Isomap(K={input_parameters[0]})": Isomap(n_components=n_components, adj_calculator=KNearestNeighbors(input_parameters[0])),
        f"Isomap(K={input_parameters[1]})": Isomap(n_components=n_components, adj_calculator=KNearestNeighbors(input_parameters[1])),
        f"Isomap(e={input_parameters[2]})": Isomap(n_components=n_components, adj_calculator=EpsNeighborhood(input_parameters[2])),
        # more n_component worse because bigger and more eigenvectors will increase the error
        f"LLE(K={input_parameters[0]})": LLE(n_components=n_components, adj_calculator=KNearestNeighbors(input_parameters[0])),
        f"LLE(K={input_parameters[1]})": LLE(n_components=n_components, adj_calculator=KNearestNeighbors(input_parameters[1])),
        f"LLE(e={input_parameters[2]})": LLE(n_components=n_components, adj_calculator=EpsNeighborhood(input_parameters[2])),
        f"Spectral(K={input_parameters[0]})": Spectral(n_components=n_components, adj_calculator=KNearestNeighbors(input_parameters[0])),
        f"Spectral(K={input_parameters[1]})": Spectral(n_components=n_components, adj_calculator=KNearestNeighbors(input_parameters[1])),
        f"Spectral(e={input_parameters[2]})": Spectral(n_components=n_components, adj_calculator=EpsNeighborhood(input_parameters[2]))
    }

    # TODO: calculate trustworthiness for each combination
    # Iterate over each algorithm and compute trustworthiness
    results = {}
    for title, algorithm in algorithms.items():
        transformed = algorithm.fit_transform(input_data)
        # Compute trustworthiness score
        T_k = trustworthiness(input_data, transformed, n_neighbors=5)
        results[title] = (transformed, T_k)
        print(f"{title} Trustworthiness: {T_k:.4f}")

    dump(results, f"{file_name}.joblib")
    return


if __name__ == "__main__":
    # TODO: load the swiss roll dataset
    # Load swiss roll dataset
    path = "datasets/swissroll.npz"
    data, labels = load_dataset(path)

    # parameters = [10, 20, 5]
    # apply_algorithms(data, parameters, "results", n_components=2)

    visualizer = Visualizer('results.joblib', labels)  # Load results and prepare the visualizer
    visualizer.plot_results()  # Plot the results for all algorithms

    # # Load ORL faces dataset
    # path = "datasets/faces.npz"
    # data2, labels2 = load_dataset(path)
    #
    # # # Apply PCA
    # num_components = 100
    # pca = PCA(n_components=num_components)
    # reduced_data = pca.fit_transform(data2)
    # # Reconstruct images using inverse_transform
    # reconstructed_data = pca.inverse_transform(reduced_data)
    # # Visualize Original vs Reconstructed Images
    # plot_images(data2, reconstructed_data)
    #
    # # Compute and plot cumulative explained variance ratio
    # cumulative_evr = np.cumsum(pca.explained_variance_ratio_)
    # plot_statistics(cumulative_evr)
    #
    # # parameters = [20, 18, 5]
    # # apply_algorithms(data2, parameters, "result_faces", n_components=num_components)
    #
    # visualizer = Visualizer('result_faces.joblib', labels2)  # Load results and prepare the visualizer
    # visualizer.plot_results()  # Plot the results for all algorithms
