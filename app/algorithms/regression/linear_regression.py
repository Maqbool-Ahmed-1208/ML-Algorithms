# import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def linear_regression(x, y, test_size = 20, random_state = 42):

    # y = bx + a   or  y = a + bx
    model = LinearRegression()
    
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size/100, random_state)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # b is slope
    b = model.coef_[0][0]
    print("Slope= ", b)

    # a is intercept
    a = model.intercept_[0]
    print("Intercept = ", a)
    
    # Accuracy score
    accuracy = accuracy_score(y_pred, y_test)
    print("Accuracy Score: ", accuracy)

    plt.scatter(x,y, color ='black', label="Actual Data")
    plt.plot(x,y_pred,color='blue', label="regression line")
    plt.xlabel("Independent variable (x)")
    plt.ylabel("Dependent variable (y)")
    plt.legend()
    plt.title("Simple Linear Regression")
    plt.show()

    return y_pred, b, a, accuracy
