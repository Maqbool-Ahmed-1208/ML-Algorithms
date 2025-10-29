# ===============================================================
# MODULE: Logistic Regression Implementation
# AUTHOR: Maqbool Ahmed
# DESCRIPTION:
#     This module demonstrates a dynamic and interpretable implementation
#     of Logistic Regression — a supervised machine learning algorithm used
#     primarily for binary and multiclass classification.
#
# THEORETICAL CONCEPTS:
# ---------------------------------------------------------------
# 1. Logistic Regression:
#    - A probabilistic model that predicts categorical outcomes using a 
#      logistic (sigmoid) function.
#    - Formula: P(Y=1|X) = 1 / (1 + e^-(β₀ + β₁X₁ + β₂X₂ + ... + βₙXₙ))
#
# 2. Cost Function (Log Loss):
#    - The model minimizes the negative log-likelihood (cross-entropy loss)
#      to estimate the coefficients (βs).
#
# 3. Evaluation Metrics:
#    - Accuracy: Overall correctness of model predictions.
#    - Precision: Proportion of true positives among predicted positives.
#    - Recall: Proportion of true positives detected out of actual positives.
#    - F1-score: Harmonic mean of precision and recall.
#
# 4. Visualization:
#    - Decision Boundary: Graphical separation of predicted classes.
#    - Confusion Matrix: Summary of prediction results showing true vs. false.
# ===============================================================

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
import pandas as pd
from sklearn.datasets import load_iris

def logistic_regression(
    X, 
    y, 
    test_size=0.2, 
    random_state=42, 
    no_iter=1000, 
    visualize=True
):
    """
    Perform Logistic Regression, visualize results, and return evaluation metrics.

    Parameters:
    ----------
    X : array-like or DataFrame
        Feature matrix.
    y : array-like or Series
        Target vector.
    test_size : float, default=0.2
        Proportion of data to be used as test set.
    random_state : int, default=42
        Random seed for reproducibility.
    no_iter : int, default=1000
        Maximum number of iterations for solver convergence.
    visualize : bool, default=True
        If True, display decision boundary and confusion matrix.

    Returns:
    -------
    results : dict
        Contains trained model, predictions, and all evaluation metrics.
    """

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Train Logistic Regression model
    model = LogisticRegression(max_iter=no_iter, random_state=random_state)
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    # Visualization
    if visualize:
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        
        # (1) Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax[0])
        ax[0].set_title('Confusion Matrix')
        ax[0].set_xlabel('Predicted')
        ax[0].set_ylabel('Actual')

        # (2) Decision Boundary (for 2D data only)
        if X.shape[1] == 2:
            x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
            y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                                 np.linspace(y_min, y_max, 200))
            Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            ax[1].contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
            ax[1].scatter(X_test.iloc[:, 0], X_test.iloc[:, 1],
                          c=y_test, edgecolor='k', cmap='coolwarm')
            ax[1].set_title('Decision Boundary')
            ax[1].set_xlabel(X.columns[0])
            ax[1].set_ylabel(X.columns[1])
        else:
            ax[1].axis('off')
            ax[1].text(0.3, 0.5, "Decision boundary only for 2D features",
                       fontsize=10, color='gray')

        plt.tight_layout()
        plt.show()

    # Display classification report
    print("\n" + "="*40)
    print("📊 CLASSIFICATION REPORT")
    print("="*40)
    print(classification_report(y_test, y_pred))

    # Prepare user-friendly results
    results = {
        "Model": model,
        "Accuracy (%)": round(accuracy * 100, 2),
        "Precision": round(precision, 3),
        "Recall": round(recall, 3),
        "F1 Score": round(f1, 3),
        "Predictions": y_pred,
    }

    print("\n✅ Model Training Complete!")
    print(f"Accuracy: {results['Accuracy (%)']}% | F1: {results['F1 Score']}\n")

    return results


    # ----------------- Example Usage -----------------

data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

results = logistic_regression(X, y)



