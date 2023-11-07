from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
import pandas as pd 
import numpy as np

 
iris =datasets.load_iris()
#divide los datasets en dos partes: training datasets y test datasets
X, y = datasets.load_iris( return_X_y = True)

# Divide 70% training y 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30) 

# Se crea el dataframe del dataset

data = pd.DataFrame({"sepallength": iris.data[:,0], "sepalwidth": iris.data[:, 1], 
                    "petallength": iris.data[:, 2], "petalwidth": iris.data[:, 3], 
                     "species": iris.target}) 

# Crea el objeto RF n_estimators es el numero de árboles que se crearán
clf= RandomForestClassifier(n_estimators=100)

# Se entrena el modelo con el training dataset
clf.fit(X_train, y_train)

# Se crean predicciones con el dataset de test
y_pred = clf.predict(X_test) 

# Se muestra la precisión
print("ACCURACY OF THE MODEL: ", metrics.accuracy_score(y_test, y_pred)) 

print(clf.predict([[3,3,2,2]]))
feature_imp = pd.Series(clf.feature_importances_, index = iris.feature_names).sort_values(ascending = False) 
print(feature_imp)



