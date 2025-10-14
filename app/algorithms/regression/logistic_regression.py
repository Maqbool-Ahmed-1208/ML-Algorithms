from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
def logistic_regression(x, y, test_size = 0.2, random_state = 42):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = test_size, random_state = random_state)

    model = LogisticRegression(tol = 0.01, max_iter = 1000)
    model.fit(X_train,y_train)

    y_pred = model.predict(X_test)
    accuracy_score(y_test, y_pred)*100
