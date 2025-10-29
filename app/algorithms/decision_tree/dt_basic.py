# =======================================================================
# MODULE NAME   : decision_tree_classifier.py
# AUTHOR        : Taj Elkatawneh
# DESCRIPTION   : 
#   This module implements a Decision Tree Classifier using scikit-learn.
#   It trains a model on labeled data, visualizes predictions, 
#   and evaluates key performance metrics.
#
# THEORETICAL CONCEPTS:
# -----------------------------------------------------------------------
# 1. Decision Tree Classifier:
#    - A supervised learning algorithm that splits data into subsets 
#      based on feature values, creating a tree-like structure of decisions.
#    - Each internal node represents a feature-based decision rule.
#    - Each leaf node represents a predicted output class.
#
# 2. Splitting Criteria:
#    - Determines how the tree decides where to split data.
#      * 'gini'    : Measures impurity using the Gini Index.
#      * 'entropy' : Measures impurity using Information Gain.
#
# 3. Overfitting:
#    - Occurs when the tree becomes too deep and captures noise.
#    - Controlled via parameters like `max_depth` or `min_samples_split`.
#
# 4. Key Metrics:
#    - Accuracy: Measures overall correctness of predictions.
#    - Depth: Indicates model complexity.
#    - Leaf Count: Total number of terminal nodes.
#    - Feature Importance: Shows which features influence decisions most.
#
# OUTPUT:
#    - Prints performance metrics.
#    - Displays actual vs predicted visualization.
#    - Returns model, predictions, and metrics in a structured dictionary.
#
# DEPENDENCIES:
#    - matplotlib
#    - scikit-learn
# =======================================================================

import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

def decision_tree(x, y, test_size=0.2, random_state=42, criterion="entropy", visualize=True):
    """
    Train a Decision Tree Classifier, visualize predictions, and return metrics.

    Parameters
    ----------
    x : array-like or DataFrame
        Feature dataset.
    y : array-like
        Target labels.
    test_size : float, optional (default=0.2)
        Fraction of data to use for testing.
    random_state : int, optional (default=42)
        Random seed for reproducibility.
    criterion : str, optional (default='entropy')
        Function to measure the quality of a split ('gini' or 'entropy').
    visualize : bool, optional (default=True)
        Whether to visualize actual vs predicted labels.

    Returns
    -------
    results : dict
        A dictionary containing:
        - 'model': Trained DecisionTreeClassifier.
        - 'accuracy': Model accuracy score.
        - 'depth': Tree depth.
        - 'num_leaves': Number of leaf nodes.
        - 'feature_importances': Feature importance values.
        - 'y_pred': Predicted labels for test set.
    """

    # ------------------------- Data Split -------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state
    )

    # ------------------------- Model Training ---------------------
    model = DecisionTreeClassifier(criterion=criterion, random_state=random_state)
    model.fit(X_train, y_train)

    # ------------------------- Prediction -------------------------
    y_pred = model.predict(X_test)

    # ------------------------- Evaluation -------------------------
    acc = accuracy_score(y_test, y_pred)
    depth = model.get_depth()
    num_leaves = model.get_n_leaves()
    importances = model.feature_importances_

    # ------------------------- Visualization ----------------------
    if visualize:
        plt.figure(figsize=(8, 5))
        plt.scatter(range(len(y_test)), y_test, color='red', label='Actual')
        plt.scatter(range(len(y_pred)), y_pred, color='blue', alpha=0.6, label='Predicted')
        plt.title(f"Decision Tree Classification ({criterion.capitalize()} Criterion)")
        plt.xlabel("Sample Index")
        plt.ylabel("Class Label")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()

    # ------------------------- Summary ----------------------------
    print("\n========== Decision Tree Summary ==========")
    print(f"Criterion Used       : {criterion}")
    print(f"Accuracy Score       : {acc:.3f}")
    print(f"Tree Depth           : {depth}")
    print(f"Number of Leaves     : {num_leaves}")
    print("Feature Importances  :", [round(val, 3) for val in importances])
    print("===========================================")

    # ------------------------- Return Results ---------------------
    results = {
        'model': model,
        'accuracy': round(acc, 3),
        'depth': depth,
        'num_leaves': num_leaves,
        'feature_importances': [round(val, 3) for val in importances],
        'y_pred': y_pred
    }

    return results

# --- Example Usage ---
# --- Generate a small classification dataset ---
X, y = make_classification(n_samples=40, n_features=4, n_informative=2,n_redundant=0, random_state=42)

# --- Test Decision Tree function ---
results = decision_tree(X, y, test_size=0.25, criterion='entropy')

print("\nReturned Results:")
print(results)
