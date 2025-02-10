import matplotlib.pyplot as plt
import numpy as np


def plot_2d_data(input_data_2d, input_labels, title):
    plt.figure(figsize=(8, 6))
    plt.scatter(input_data_2d[:, 0], input_data_2d[:, 1], c=input_labels, cmap='Spectral', s=15)
    plt.colorbar(label="Labels")
    plt.title(title)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.show()


def plot_3d_data(input_data, input_labels, title):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(input_data[:, 0], input_data[:, 1], input_data[:, 2], c=input_labels, cmap='Spectral', s=15)
    legend1 = ax.legend(*scatter.legend_elements(), title="Labels")
    ax.scatter(np.mean(input_data, axis=0)[0], np.mean(input_data, axis=0)[1], np.mean(input_data, axis=0)[2], marker="x", c="red")
    ax.add_artist(legend1)
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()


def plot_images(data, reconstructed_data):
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    for i in range(5):
        axes[0, i].imshow(data[i].reshape(32, 32), cmap='gray')
        axes[0, i].set_title("Original")
        axes[0, i].axis("off")

        axes[1, i].imshow(reconstructed_data[i].reshape(32, 32), cmap='gray')
        axes[1, i].set_title("Reconstructed")
        axes[1, i].axis("off")
    plt.show()


def plot_statistics(cumulative_evr):
    plt.figure(figsize=(8, 6))
    plt.plot(cumulative_evr, marker='o', linestyle='-')
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance Ratio")
    plt.title("Explained Variance Ratio for ORL Faces")
    plt.grid()
    plt.show()
