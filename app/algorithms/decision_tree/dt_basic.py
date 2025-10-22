import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def decision_tree(x, y, test_size=0.2, random_state=42, criterion="entropy"):
    model = DecisionTreeClassifier(criterion=criterion, random_state=random_state)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy =", acc)
    plt.scatter(range(len(y_test)), y_test, color='red', label='Actual')
    plt.scatter(range(len(y_pred)), y_pred, color='blue', alpha=0.6, label='Predicted')
    plt.title("Decision Tree Classification")
    plt.xlabel("Samples")
    plt.ylabel("Class (0 or 1)")
    plt.legend()
    plt.show()
    return y_pred, acc, model