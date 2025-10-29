"""
===============================================================================
MODULE: linear_regression_model.py
AUTHOR: Maqbool Ahmed
DESCRIPTION:
    This module implements a Simple/Multi Linear Regression model using 
    scikit-learn. It performs the following tasks:
      1. Splits dataset into training and testing sets.
      2. Trains a Linear Regression model.
      3. Computes key performance metrics (MSE, RMSE, R²).
      4. Visualizes regression fit using matplotlib.
      5. Returns user-friendly structured results.

THEORETICAL CONCEPTS:
    - LINEAR REGRESSION EQUATION:
        y = a + bX
        where:
            a = Intercept (constant term)
            b = Slope (coefficient of independent variable X)
    - GOAL:
        Find the best-fitting line that minimizes the residual sum of squares
        (difference between actual and predicted values).
    - METRICS:
        MSE  (Mean Squared Error): Measures average squared difference.
        RMSE (Root Mean Squared Error): Square root of MSE for interpretability.
        R²   (Coefficient of Determination): Proportion of variance explained.

DEPENDENCIES:
    - numpy
    - pandas
    - matplotlib
    - scikit-learn
===============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def linear_regression(X, y, test_size=0.2, random_state=42, visualize=True):
    """
    Perform Linear Regression on given data and visualize results.

    Parameters
    ----------
    X : array-like or DataFrame
        Independent variable(s)
    y : array-like or Series
        Dependent variable
    test_size : float, optional
        Fraction of data to be used for testing (default=0.2)
    random_state : int, optional
        Controls train-test split reproducibility (default=42)
    visualize : bool, optional
        If True, displays the regression visualization

    Returns
    -------
    results : dict
        Dictionary containing model, coefficients, metrics, and predictions
    """

    # Ensure correct input shape
    if len(np.array(X).shape) == 1:
        X = np.array(X).reshape(-1, 1)
    if len(np.array(y).shape) == 1:
        y = np.array(y).reshape(-1, 1)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Extract model parameters
    slope = model.coef_.flatten()
    intercept = model.intercept_[0]

    # Compute metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # Visualization (for 1D only)
    if visualize and X.shape[1] == 1:
        plt.figure(figsize=(8, 5))
        plt.scatter(X_train, y_train, color='black', label="Training Data", alpha=0.7)
        plt.scatter(X_test, y_test, color='red', label="Test Data", alpha=0.7)
        plt.plot(X_test, y_pred, color='blue', linewidth=2, label="Regression Line")
        plt.xlabel("Independent Variable (X)")
        plt.ylabel("Dependent Variable (Y)")
        plt.title("Linear Regression Fit")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.show()

    # User-friendly summary
    results = {
        "Model": model,
        "Slope (b)": slope.tolist(),
        "Intercept (a)": intercept,
        "Mean Squared Error (MSE)": round(mse, 4),
        "Root Mean Squared Error (RMSE)": round(rmse, 4),
        "R² Score": round(r2, 4),
        "Predictions": y_pred.flatten().tolist()
    }

    return results



# ------------- Example Usage ------------- 

# Example Data
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([2, 4, 5, 4, 5]).reshape(-1, 1)

results = linear_regression(X, y)
print(results)
