# import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def linear_regression(x, y, test_size = 0.2, random_state = 42):

    # y = bx + a   or  y = a + bx
    model = LinearRegression()
    
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = test_size, random_state = random_state)


    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # print(len(y_pred),len(y_test))
    print(len(X_train),len(y_train))
    # b is slope
    b = model.coef_[0][0]
    print("Slope= ", b)

    # a is intercept
    a = model.intercept_[0]
    print("Intercept = ", a)
    
    mse = mean_squared_error(y_test, y_pred)
    print("mean_squared_error", mse)

    plt.scatter(X_train, y_train, color='black', label="Training Data")
    plt.scatter(X_test, y_test, color='red', label="Test Data")
    plt.plot(X_test, y_pred, color='blue', label="Regression Line")

    plt.xlabel("Independent Variable (X)")
    plt.ylabel("Dependent Variable (Y)")
    plt.title("Simple Linear Regression")
    plt.legend()
    plt.show()

    return y_pred, b, a, mse
