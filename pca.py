import numpy as np
from matplotlib import pyplot as plt
from dataset import load_dataset


class PCA:
    """
    Principal Component Analysis (PCA) for dimensionality reduction.
    """

    def __init__(self, n_components):
        """
        Initialize PCA with the number of components to retain.

        Parameters:
        - n_components: int, the number of principal components to keep.
        """
        # TODO: initialize required instance variables.
        self.n_components = n_components
        self.top_eigenvalues, self.top_eigenvectors = None, None
        self.eigenvalues, self.eigenvectors = None, None
        self.average = None
        self.explained_variance_ratio_ = None

    def _center_data(self, X):
        # TODO: Compute the mean of X along axis 0 (features) and subtract it from X
        self.average = np.mean(X, axis=0)
        X_centered = X - self.average
        return X_centered

    def _create_cov(self, X):
        # TODO: Use the formula for the covariance matrix.
        # number of samples
        n = X.shape[0]
        # Covariance formula: (1 / n-1) X^T X
        cov_matrix = (1/(n-1)) * (X.T@X)


        # # am I supposed to use np.cov or not?
        # # is it n or n-1?
        # cov_matrix1 = np.cov(X, rowvar=False)
        # print(np.allclose(cov_matrix1, cov_matrix, atol=1e-2))
        # print(cov_matrix1)
        # print(cov_matrix)


        return cov_matrix

    def _decompose(self, covariance_matrix):
        # TODO: Use np.linalg.eig to get eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
        # Sort eigenvalues and eigenvectors in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]
        self.eigenvalues = eigenvalues[sorted_indices]
        self.eigenvectors = eigenvectors[:, sorted_indices]

        # # Align eigenvectors with sklearn's PCA (flip if necessary)
        # # if v is an eigenvector so is -v
        # eigenvectors[:, eigenvectors[0, :] < 0] *= -1
        return

    def fit(self, X):
        """
        Fit the PCA model to the dataset by computing the principal components.

        Parameters:
        - X: numpy array, the centered dataset (m x n).
        """
        # TODO: Center the data
        X_centered = self._center_data(X)
        # TODO: Compute the covariance matrix
        cov_matrix = self._create_cov(X_centered)
        # TODO: Perform eigen decomposition
        self._decompose(cov_matrix)
        # get top principal components
        self.top_eigenvectors = self.eigenvectors[:, :self.n_components]
        self.top_eigenvalues = self.eigenvalues[:self.n_components]

        # should I return anything? or should I do anything else?
        return
    
    def transform(self, X):
        """
        Project the data onto the top principal components.

        Parameters:
        - X: numpy array, the data to project (m x n).

        Returns:
        - transformed_data: numpy array, the data projected onto the top principal components.
        """
        # TODO: Center the data
        X_centered = self._center_data(X)
        # TODO: Apply projection
        transformed_data = X_centered @ self.top_eigenvectors
        # Compute explained variance ratio (EVR)
        self.explained_variance_ratio_ = self.eigenvalues / np.sum(self.eigenvalues)
        return transformed_data

    def fit_transform(self, X):
        """
        Fit the PCA model and transform the data in one step.

        Parameters:
        - X: numpy array, the data to fit and transform (m x n).

        Returns:
        - transformed_data: numpy array, the data projected onto the top principal components.
        """
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_transformed):
        """
        Reconstruct the original data from the transformed data.

        Parameters:
        - X_transformed: numpy array, the data in the reduced dimensional space.

        Returns:
        - original_data: numpy array, the reconstructed data in the original space.
        """
        # TODO: Apply reconstruction formula
        X = self.average + (X_transformed @ self.top_eigenvectors.T)

        return X


if __name__ == "__main__":
    # TODO: Load swiss roll dataset
    path = "datasets/swissroll.npz"
    data, labels = load_dataset(path)

    # Visualize the Swiss Roll in 3D
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=labels, cmap='Spectral', s=15)
    legend1 = ax.legend(*scatter.legend_elements(), title="Labels")
    ax.add_artist(legend1)
    ax.set_title("Swiss Roll Dataset")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()

    # TODO: Perform PCA
    # Apply PCA to reduce to 2D
    pca = PCA(n_components=2)
    data_2d = pca.fit_transform(data)

    from sklearn.decomposition import PCA as SklearnPCA
    # Compare PCA
    sklearn_pca = SklearnPCA(n_components=2)
    sklearn_transformed = sklearn_pca.fit_transform(data)
    pca_error = np.linalg.norm(data_2d - sklearn_transformed)
    print(f"PCA Error: {pca_error:.2f}")


    # TODO: Visualize the results
    # Visualize the 2D projection
    plt.figure(figsize=(8, 6))
    plt.scatter(data_2d[:, 0], data_2d[:, 1], c=labels, cmap='Spectral', s=15)
    plt.colorbar(label="Labels")
    plt.title("PCA Projection of Swiss Roll")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.show()

    # TODO: Reconstruct dataset
    # Reconstruct the data
    reconstructed_data = pca.inverse_transform(data_2d)
    # Calculate and display reconstruction error
    reconstruction_error = np.linalg.norm(data - reconstructed_data)
    print(f"Reconstruction Error: {reconstruction_error:.2f}")


    # # Generate a 2D plane in 3D space with 4 classes
    # data, labels = generate_plane(d=2, dim=3, n_samples=500, classes=4, noise_std=0.5)
    #
    # # Plot the data in 3D
    # fig = plt.figure(figsize=(10, 8))
    # ax = fig.add_subplot(111, projection='3d')
    # scatter = ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=labels, cmap='viridis', s=15)
    # legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
    # ax.add_artist(legend1)
    # ax.scatter(np.mean(data, axis=0)[0], np.mean(data, axis=0)[1], np.mean(data, axis=0)[2], marker="x", c="red")
    # ax.set_title("2D Hyperplane in 3D Space")
    # ax.set_xlabel("X")
    # ax.set_ylabel("Y")
    # ax.set_zlabel("Z")
    # plt.show()
    #
    # # Perform PCA
    # pca = PCA(n_components=2)
    # data_2d = pca.fit_transform(data)
    #
    # # Visualize the projected data
    # plt.figure(figsize=(8, 6))
    # plt.scatter(data_2d[:, 0], data_2d[:, 1], c=labels, cmap='viridis', s=15)
    # plt.colorbar(label="Labels")
    # plt.title("PCA Projection of 2D Hyperplane")
    # plt.xlabel("Principal Component 1")
    # plt.ylabel("Principal Component 2")
    # plt.show()
