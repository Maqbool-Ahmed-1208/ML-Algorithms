# =======================================================================
# MODULE NAME   : agglomerative_clustering.py
# AUTHOR        : Taj Elkatawneh
# DESCRIPTION   : 
#   This module performs hierarchical (agglomerative) clustering 
#   on a given dataset, visualizes the resulting dendrogram, 
#   and computes relevant clustering metrics for interpretation.
#
# THEORETICAL CONCEPTS:
# -----------------------------------------------------------------------
# 1. Hierarchical Clustering:
#    - A clustering approach that builds a hierarchy of clusters by 
#      successively merging or splitting them based on similarity or distance.
#    - The Agglomerative approach (used here) starts with each data point
#      as its own cluster and merges them step by step.
#
# 2. Linkage Methods:
#    - Define how the distance between clusters is computed.
#      * 'single'   : Minimum distance between clusters.
#      * 'complete' : Maximum distance between clusters.
#      * 'average'  : Average pairwise distance between all points.
#      * 'ward'     : Minimizes the total within-cluster variance.
#
# 3. Dendrogram:
#    - A visual representation of the cluster hierarchy.
#    - Each merge is shown as a horizontal line, with the height representing
#      the distance (dissimilarity) between clusters being merged.
#
# 4. Clustering Metrics:
#    - Number of merges   : Total cluster combinations performed.
#    - Maximum distance    : Largest linkage distance (indicating final merge).
#    - Mean distance       : Average linkage distance across all merges.
#    - Cophenetic Correlation Coefficient (CCC):
#        Measures how well the dendrogram preserves pairwise distances.
#        Values closer to 1 indicate better clustering structure.
#
# OUTPUT:
#    - A matplotlib dendrogram plot.
#    - A summary of key clustering metrics printed and returned.
#
# DEPENDENCIES:
#    - numpy
#    - matplotlib
#    - scipy
#
# =======================================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, cophenet
from scipy.spatial.distance import pdist
import pandas as pd
from sklearn.datasets import make_blobs

def hierarchical_clustering(df, method='ward', metric='euclidean', title='Hierarchical Clustering Dendrogram'):
    """
    Perform hierarchical clustering and visualize results.

    Parameters
    ----------
    df : pandas.DataFrame or numpy.ndarray
        Input dataset containing features for clustering.
    method : str, optional (default='ward')
        Linkage method ('single', 'complete', 'average', 'ward').
    metric : str, optional (default='euclidean')
        Distance metric used to compute pairwise distances.
    title : str, optional
        Title for the dendrogram visualization.

    Returns
    -------
    results : dict
        A dictionary containing:
        - 'linkage_matrix': Computed linkage matrix.
        - 'num_merges'    : Number of cluster merges.
        - 'max_distance'  : Maximum linkage distance.
        - 'mean_distance' : Mean linkage distance.
        - 'cophenetic_corr': Cophenetic correlation coefficient.
    """

    # Compute hierarchical linkage matrix using the specified method and metric
    linkage_matrix = linkage(df, method=method, metric=metric)

    # Compute pairwise distances and cophenetic correlation coefficient
    distances = pdist(df, metric=metric)
    coph_corr, _ = cophenet(linkage_matrix, distances)

    # Calculate relevant clustering metrics
    num_merges = linkage_matrix.shape[0]
    max_distance = np.max(linkage_matrix[:, 2])
    mean_distance = np.mean(linkage_matrix[:, 2])

    # --- Visualization using Matplotlib ---
    plt.figure(figsize=(10, 6))
    dendrogram(
        linkage_matrix,
        orientation='top',
        distance_sort='ascending',
        no_labels=False,
        color_threshold=max_distance * 0.7  # dynamic color cut
    )
    plt.title(f"{title} ({method.capitalize()} Linkage)")
    plt.xlabel("Sample Index")
    plt.ylabel("Distance")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    # --- Prepare user-friendly results ---
    print("\n========== Hierarchical Clustering Summary ==========")
    print(f"Linkage Method          : {method}")
    print(f"Distance Metric         : {metric}")
    print(f"Number of Merges        : {num_merges}")
    print(f"Maximum Distance        : {max_distance:.3f}")
    print(f"Mean Distance           : {mean_distance:.3f}")
    print(f"Cophenetic Corr. Coef.  : {coph_corr:.3f}")
    print("=====================================================")

    results = {
        'linkage_matrix': linkage_matrix,
        'num_merges': num_merges,
        'max_distance': round(max_distance, 3),
        'mean_distance': round(mean_distance, 3),
        'cophenetic_corr': round(coph_corr, 3)
    }

    return results

# --- Example Usage ---

# --- Generate synthetic data ---
X, _ = make_blobs(n_samples=20, centers=3, n_features=2, random_state=42)
df = pd.DataFrame(X, columns=['Feature1', 'Feature2'])

# --- Test function ---
results = hierarchical_clustering(df, method='ward', title='Test Hierarchical Clustering')

print("\nReturned Results:")
print(results)
