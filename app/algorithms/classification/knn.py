from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd


def KNN(X, Y, N, Test_size):

  X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = Test_size, random_state = 42)

  knn = KNeighborsClassifier(n_neighbors = N)
  knn.fit(X_train, y_train)
  y_pred = knn.predict(X_test)
  y_pred = pd.DataFrame({"prediction" :  y_pred})
  # print("Accuracy Score: ",accuracy_score(y_test, y_pred)*100)
  acc = accuracy_score(y_test, y_pred)*100

  return y_pred, acc