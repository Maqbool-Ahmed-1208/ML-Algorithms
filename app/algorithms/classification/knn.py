"""
===============================================================================
K-NEAREST NEIGHBORS (KNN) CLASSIFIER
===============================================================================
Author      : Maqbool Ahmed
Module      : knn_classifier.py
Description : Implementation of the K-Nearest Neighbors algorithm for 
              supervised classification tasks with evaluation metrics and 
              visualization.

---------------------------
THEORETICAL CONCEPT:
---------------------------
1. **Definition:**
   K-Nearest Neighbors (KNN) is a *non-parametric, instance-based learning* 
   algorithm used for both classification and regression. It assumes that 
   similar points exist in close proximity in the feature space.

2. **Working Principle:**
   - The algorithm calculates the *distance* (commonly Euclidean distance) 
     between the query point and all points in the training dataset.
   - It selects the 'K' closest samples.
   - For classification, it assigns the class label most frequent among these 
     neighbors (majority voting).

3. **Characteristics:**
   - *Lazy learner:* No explicit training phase; the computation occurs at 
     prediction time.
   - *Distance-based:* Sensitive to the scale of features — normalization or 
     standardization is often required.
   - *Parameter K:* 
       - Small K → high variance (overfitting)
       - Large K → high bias (underfitting)

4. **Applications:**
   - Pattern recognition
   - Recommendation systems
   - Medical diagnosis
   - Anomaly detection

---------------------------
EVALUATION METRICS:
---------------------------
- **Accuracy:** Percentage of correctly predicted samples.
- **Precision:** Proportion of positive identifications that were actually correct.
- **Recall (Sensitivity):** Proportion of actual positives identified correctly.
- **F1 Score:** Harmonic mean of precision and recall.
- **Confusion Matrix:** Table showing true vs predicted classifications.

---------------------------
VISUALIZATION:
---------------------------
A confusion matrix heatmap is plotted using Matplotlib and Seaborn to 
visualize the model’s prediction performance.

===============================================================================
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    precision_score, recall_score, f1_score
)
from sklearn.neighbors import KNeighborsClassifier


def knn_classifier(X, Y, n_neighbors=5, test_size=0.2, random_state=42, show_plot=True):
    """
    Train and evaluate a K-Nearest Neighbors (KNN) classifier.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Feature matrix.
    Y : pd.Series or np.ndarray
        Target labels.
    n_neighbors : int, default=5
        Number of neighbors (K) used for classification.
    test_size : float, default=0.2
        Proportion of the dataset reserved for testing.
    random_state : int, default=42
        Random seed for reproducibility.
    show_plot : bool, default=True
        Whether to display a confusion matrix heatmap.

    Returns
    -------
    results : dict
        A dictionary containing:
        - summary: accuracy, precision, recall, f1-score (in %)
        - classification_report: detailed sklearn report
        - confusion_matrix: numerical matrix of true vs predicted values
        - predictions: DataFrame with actual and predicted labels
    """

    # ---------------------------
    # STEP 1: Split the dataset
    # ---------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=test_size, random_state=random_state
    )

    # ---------------------------
    # STEP 2: Initialize the model
    # ---------------------------
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)

    # ---------------------------
    # STEP 3: Fit the model
    # ---------------------------
    knn.fit(X_train, y_train)

    # ---------------------------
    # STEP 4: Make predictions
    # ---------------------------
    y_pred = knn.predict(X_test)

    # ---------------------------
    # STEP 5: Compute evaluation metrics
    # ---------------------------
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=False)

    # ---------------------------
    # STEP 6: Visualization
    # ---------------------------
    if show_plot:
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm', cbar=False)
        plt.title(f"KNN Confusion Matrix (k={n_neighbors})", fontsize=12)
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.tight_layout()
        plt.show()

    # ---------------------------
    # STEP 7: Prepare user-friendly summary
    # ---------------------------
    summary = {
        "Accuracy (%)": round(acc * 100, 2),
        "Precision (%)": round(precision * 100, 2),
        "Recall (%)": round(recall * 100, 2),
        "F1 Score (%)": round(f1 * 100, 2)
    }

    # ---------------------------
    # STEP 8: Return structured results
    # ---------------------------
    results = {
        "summary": summary,
        "classification_report": report,
        "confusion_matrix": cm,
        "predictions": pd.DataFrame({"Actual": y_test, "Predicted": y_pred}).reset_index(drop=True)
    }

    return results

# =============================================================================
# EXAMPLE USAGE: KNN CLASSIFIER
# =============================================================================

# Import libraries
from sklearn.datasets import load_iris

# Import our KNN function
# from knn_classifier import knn_classifier   # Uncomment if using as separate file

# Load example dataset
iris = load_iris()

# Prepare data
X = pd.DataFrame(iris.data, columns=iris.feature_names)
Y = pd.Series(iris.target)

# Display dataset info
print("🔹 Dataset Shape:", X.shape)
print("🔹 Feature Names:", iris.feature_names)
print("🔹 Target Classes:", iris.target_names)

# -----------------------------------------------------------------------------
# Run the KNN Classifier
# -----------------------------------------------------------------------------
results = knn_classifier(
    X, Y, 
    n_neighbors=5,       # Number of nearest neighbors
    test_size=0.3,       # 30% of data for testing
    random_state=42,     # Reproducibility
    show_plot=True       # Display confusion matrix
)

# -----------------------------------------------------------------------------
# View Model Results
# -----------------------------------------------------------------------------

print("\n📊 MODEL PERFORMANCE SUMMARY:")
for metric, value in results["summary"].items():
    print(f"{metric:<15}: {value}")

print("\n🧾 DETAILED CLASSIFICATION REPORT:")
print(results["classification_report"])

print("\n🔍 SAMPLE PREDICTIONS:")
print(results["predictions"].head())
