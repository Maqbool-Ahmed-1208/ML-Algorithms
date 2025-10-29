# =======================================================================
# MODULE NAME   : kmeans.py
# AUTHOR        : Taj Elkatawneh
# DESCRIPTION   : 
#   Custom implementation of the K-Means clustering algorithm from scratch.
#   Includes centroid initialization, iterative optimization, 
#   cluster visualization, and evaluation metrics.
#
# THEORETICAL CONCEPTS:
# -----------------------------------------------------------------------
# 1. K-Means Clustering:
#    - An unsupervised machine learning algorithm that partitions data 
#      into K clusters by minimizing intra-cluster variance (SSE).
#    - Each cluster is represented by its centroid (mean position).
#
# 2. Algorithm Steps:
#    (a) Initialize K random centroids.
#    (b) Assign each data point to the nearest centroid (Euclidean distance).
#    (c) Recalculate centroids based on the mean of assigned points.
#    (d) Repeat until centroids stabilize or max iterations are reached.
#
# 3. Key Terms:
#    - Centroid: Mean position of all points in a cluster.
#    - Inertia / SSE (Sum of Squared Errors): 
#        Measures compactness of clusters. Lower = better clustering.
#
# 4. Metrics Used:
#    - Inertia (SSE)
#    - Number of iterations until convergence
#    - Final cluster centroids
#
# OUTPUT:
#    - Matplotlib visualization of clusters and centroids.
#    - Returns clustering metrics and results in a user-friendly dictionary.
#
# DEPENDENCIES:
#    - numpy
#    - matplotlib
# =======================================================================

import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs

# ---------------------------- Utility Function -------------------------
def euclidean_distance(a, b):
    """Compute Euclidean distance between two points."""
    return np.sqrt(np.sum((a - b) ** 2))


# ---------------------------- KMeans Class -----------------------------
class KMeans:
    def __init__(self, k=3, max_iters=100, random_state=42):
        """
        Initialize KMeans parameters.

        Parameters:
        -----------
        k : int
            Number of clusters.
        max_iters : int
            Maximum iterations for convergence.
        random_state : int
            Random seed for reproducibility.
        """
        self.k = k
        self.max_iters = max_iters
        self.random_state = random_state
        self.centroids = None
        self.labels = None
        self.inertia_ = None
        self.iterations_ = 0

    # --------------------------- Fit Method ----------------------------
    def fit(self, X):
        """Run the KMeans clustering algorithm."""
        np.random.seed(self.random_state)
        random_indices = np.random.choice(X.shape[0], self.k, replace=False)
        self.centroids = X[random_indices]

        for i in range(self.max_iters):
            self.labels = self._assign_clusters(X)
            new_centroids = self._calculate_centroids(X)

            # Check for convergence
            if np.allclose(self.centroids, new_centroids):
                self.iterations_ = i + 1
                break

            self.centroids = new_centroids
        else:
            self.iterations_ = self.max_iters

        # Calculate final SSE (inertia)
        self.inertia_ = self._calculate_inertia(X)

        return self._get_results()

    # -------------------------- Assign Clusters ------------------------
    def _assign_clusters(self, X):
        """Assign each point to the nearest centroid."""
        labels = []
        for point in X:
            distances = [euclidean_distance(point, centroid) for centroid in self.centroids]
            labels.append(np.argmin(distances))
        return np.array(labels)

    # -------------------------- Update Centroids -----------------------
    def _calculate_centroids(self, X):
        """Compute new centroids as mean of assigned cluster points."""
        centroids = []
        for i in range(self.k):
            cluster_points = X[self.labels == i]
            centroids.append(np.mean(cluster_points, axis=0))
        return np.array(centroids)

    # -------------------------- Compute Inertia ------------------------
    def _calculate_inertia(self, X):
        """Compute the sum of squared distances (SSE)."""
        sse = 0
        for i, centroid in enumerate(self.centroids):
            cluster_points = X[self.labels == i]
            sse += np.sum((cluster_points - centroid) ** 2)
        return sse

    # --------------------------- Prediction ----------------------------
    def predict(self, X):
        """Assign new data points to the nearest cluster."""
        return self._assign_clusters(X)

    # --------------------------- Visualization -------------------------
    def plot_clusters(self, X):
        """Visualize clustered data and centroids (2D only)."""
        if X.shape[1] != 2:
            print("Visualization only supported for 2D data.")
            return

        plt.figure(figsize=(8, 6))
        for i in range(self.k):
            cluster_points = X[self.labels == i]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {i+1}', s=50)
        plt.scatter(self.centroids[:, 0], self.centroids[:, 1], c='black', marker='X', s=200, label='Centroids')
        plt.title(f'K-Means Clustering (k={self.k})')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()

    # --------------------------- Results -------------------------------
    def _get_results(self):
        """Return results in a user-friendly format."""
        results = {
            'centroids': np.round(self.centroids, 3),
            'labels': self.labels,
            'inertia': round(self.inertia_, 3),
            'iterations': self.iterations_,
        }

        print("\n========== K-Means Clustering Summary ==========")
        print(f"Number of Clusters (k) : {self.k}")
        print(f"Iterations Until Convergence: {self.iterations_}")
        print(f"Inertia (SSE)          : {self.inertia_:.3f}")
        print("Final Centroids:\n", np.round(self.centroids, 3))
        print("=================================================")
        return results



# --- Example Usage ---
# --- Generate simple 2D data ---
X, _ = make_blobs(n_samples=30, centers=3, n_features=2, random_state=42)

# --- Initialize and fit model ---
kmeans = KMeans(k=3, max_iters=100)
results = kmeans.fit(X)
kmeans.plot_clusters(X)

print("\nReturned Results:")
print(results)
