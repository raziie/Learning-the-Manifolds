import matplotlib.pyplot as plt
from collections import defaultdict
import re
from joblib import load


class Visualizer:
    def __init__(self, filename, labels):
        """
        Initializes the Visualizer with the provided results file.
        :param filename: Path to the results file (e.g., 'results.joblib')
        """
        self.labels = labels
        self.results = load(filename)  # Load results
        self.grouped_results = self.group_results_by_algorithm()  # Group results by algorithm

    def group_results_by_algorithm(self):
        """
        Groups results by the base algorithm name (ignores parameters like K=10).
        :return: A dictionary grouping results by base algorithm name
        """
        grouped_results = defaultdict(list)
        for name, (transformed_data, trust) in self.results.items():
            # Extract the base name (ignores parameters in parentheses)
            base_name = re.sub(r"\(.*\)", "", name).strip()
            grouped_results[base_name].append((name, transformed_data, trust))
        return grouped_results

    def plot_results(self):
        """
        Plots the results for each algorithm (grouped by base name) in subplots.
        """
        for base_name, group in self.grouped_results.items():
            self.plot_algorithm_results(base_name, group)

    def plot_algorithm_results(self, base_name, group):
        """
        Creates a figure for each base algorithm and plots variations in subplots.
        :param base_name: The base algorithm name (e.g., 'Isomap', 'LLE')
        :param group: The list of results related to that base algorithm
        """
        n_axes = len(group)  # Number of variations (subplots)
        n_cols = 3  # Number of columns for subplots
        n_rows = (n_axes + n_cols - 1) // n_cols  # Calculate rows based on the number of subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten()  # Flatten axes for easy iteration

        # Plot each variation in its own subplot
        for i, (name, transformed_data, trust) in enumerate(group):
            ax = axes[i]
            ax.scatter(transformed_data[:, 0], transformed_data[:, 1], c=self.labels, cmap="Spectral", s=10)
            ax.set_title(f"{name} - Trust: {trust:.4f}")
            ax.set_xlabel("Component 1")
            ax.set_ylabel("Component 2")
            ax.set_xticks([])  # Remove x-axis ticks
            ax.set_yticks([])  # Remove y-axis ticks

        # Hide unused subplots (if any)
        for j in range(n_axes, len(axes)):
            axes[j].axis("off")

        # Add a title for the whole figure and adjust layout
        fig.suptitle(f"{base_name}", fontsize=16)
        plt.tight_layout(rect=(0.0, 0.03, 1.0, 0.95))  # Adjust for subtitle
        plt.show()
