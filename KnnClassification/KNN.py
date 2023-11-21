from sklearn import datasets
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn

iris = datasets.load_iris()

df = pd.DataFrame(iris.data, columns=iris.feature_names)

print(df.head())

# Se crea el conjunto de datos de test
X = df
X = X.values
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
# Se crea el clasificador KNN
knn = KNeighborsClassifier(n_neighbors=120)
knn.fit(X_train, y_train)

print(knn.score(X_train, y_train))

print(knn.predict([[4.8, 3.0, 1.5, 0.3]]))

y_pred = knn.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(7, 5))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')

plt.show()
