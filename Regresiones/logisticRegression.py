import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

X=np.array([3.78, 2.44, 2.09, 0.14, 1.72, 1.65, 4.92, 4.37, 4.96, 4.52, 3.69, 5.88]).reshape(-1,1)
y=np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

logr=linear_model.LogisticRegression()
logr.fit(X,y)



X_test = np.linspace(X.min(),X.max(),100).reshape(-1,1)
# Para elegir una de las 2 lineas [:,0] [:,1]
y_pred = logr.predict_proba(X_test)

# Muestra los puntos
#plt.scatter(X,y)
plt.scatter(X, y, color='blue', label='Datos')
# Muestra la recta
plt.plot(X_test, y_pred, color='red', linewidth=3, label='Regresión Logística')
#plt.plot(X_test,y_pred)


plt.xlabel('X')
plt.ylabel('Etiquetas (0 o 1)')
plt.title('Regresión Logística')
plt.legend(loc="best")

# Muestra todo
plt.show()
