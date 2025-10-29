# =======================================================================
# MODULE NAME   : mini_batch_kmeans_clustering.py
# AUTHOR        : Taj Elkatawneh
# DESCRIPTION   : 
#   This module applies the Mini-Batch K-Means algorithm for scalable 
#   clustering on large datasets. It includes automatic preprocessing, 
#   visualization, and key clustering metrics for evaluation.
#
# THEORETICAL CONCEPTS:
# -----------------------------------------------------------------------
# 1. Mini-Batch K-Means:
#    - A variant of K-Means that uses small random subsets ("mini-batches") 
#      of data to update centroids iteratively, making it faster and 
#      more efficient for large datasets.
#    - Reduces computation time while maintaining similar clustering quality.
#
# 2. Algorithm Workflow:
#    (a) Initialize K centroids randomly.
#    (b) Draw small random batches of data points.
#    (c) Assign points in each batch to the nearest centroid.
#    (d) Update centroids incrementally using each mini-batch.
#
# 3. Key Terms:
#    - Batch Size: Number of samples processed per iteration.
#    - Inertia (SSE): Sum of squared distances between data points and 
#      their assigned cluster centroids (lower = better).
#    - Iterations: Number of optimization cycles before convergence.
#
# 4. Advantages:
#    - Faster and memory-efficient compared to standard K-Means.
#    - Suitable for streaming or large-scale data.
#
# 5. Visualization:
#    - Displays a scatter plot of clustered data (if 2D).
#
# OUTPUT:
#    - Prints clustering summary metrics.
#    - Returns a user-friendly dictionary with clustering results.
#
# DEPENDENCIES:
#    - numpy
#    - pandas
#    - matplotlib
#    - scikit-learn
# =======================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs

def mini_batch_kmeans(df, no_clusters=3, batch_size=32, random_state=42, scale_data=True, visualize=True):
    """
    Perform Mini-Batch K-Means clustering and visualize results.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset containing numerical features for clustering.
    no_clusters : int, optional (default=3)
        Number of clusters to form.
    batch_size : int, optional (default=32)
        Number of samples per mini-batch.
    random_state : int, optional (default=42)
        Random seed for reproducibility.
    scale_data : bool, optional (default=True)
        Whether to standardize the dataset before clustering.
    visualize : bool, optional (default=True)
        Whether to display a 2D scatter plot (for 2D data only).

    Returns
    -------
    results : dict
        A dictionary containing:
        - 'centroids': Final cluster centroids.
        - 'labels': Cluster labels assigned to each sample.
        - 'inertia': Sum of squared distances (SSE).
        - 'iterations': Number of iterations run.
        - 'batch_size': Batch size used in training.
    """

    # --- Data Preparation ---
    data = df.copy()
    if scale_data:
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
    else:
        data_scaled = data.values

    # --- Mini-Batch KMeans Initialization ---
    mbkmeans = MiniBatchKMeans(
        n_clusters=no_clusters,
        batch_size=batch_size,
        random_state=random_state,
        n_init=10
    )

    # --- Fit and Predict Clusters ---
    labels = mbkmeans.fit_predict(data_scaled)
    data['Cluster'] = labels

    # --- Metrics Calculation ---
    inertia = mbkmeans.inertia_
    iterations = mbkmeans.n_iter_

    # --- Visualization (for 2D only) ---
    if visualize and data.shape[1] - 1 == 2:
        plt.figure(figsize=(8, 6))
        plt.scatter(data.iloc[:, 0], data.iloc[:, 1], c=data['Cluster'], cmap='rainbow', s=50)
        plt.scatter(
            mbkmeans.cluster_centers_[:, 0],
            mbkmeans.cluster_centers_[:, 1],
            c='black', marker='X', s=200, label='Centroids'
        )
        plt.title(f"Mini-Batch K-Means Clustering (k={no_clusters})")
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()
    elif visualize:
        print("Visualization skipped (only supported for 2D datasets).")

    # --- Print Summary ---
    print("\n========== Mini-Batch K-Means Summary ==========")
    print(f"Number of Clusters : {no_clusters}")
    print(f"Batch Size         : {batch_size}")
    print(f"Iterations         : {iterations}")
    print(f"Inertia (SSE)      : {inertia:.3f}")
    print("Centroids:\n", np.round(mbkmeans.cluster_centers_, 3))
    print("================================================")

    # --- Return Results ---
    results = {
        'centroids': np.round(mbkmeans.cluster_centers_, 3),
        'labels': labels,
        'inertia': round(inertia, 3),
        'iterations': iterations,
        'batch_size': batch_size
    }

    return results


# --- Example Usage ---
# --- Create sample dataset ---
X, _ = make_blobs(n_samples=50, centers=3, n_features=2, random_state=42)
df = pd.DataFrame(X, columns=['Feature1', 'Feature2'])

# --- Run Mini-Batch K-Means ---
results = mini_batch_kmeans(df, no_clusters=3, batch_size=10, random_state=42)

print("\nReturned Results:")
print(results)

