import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression

# Carga los datos desde un archivo CSV a un DataFrame
data = pd.read_csv('data.csv')

# Extrae las columnas necesarias para la regresión lineal (X e y)
X = data[['Volume', 'Weight']]
y = data['CO2']

# Ajusta un modelo de regresión lineal
regr= LinearRegression()
regr.fit(X, y)

warr= np.linspace(X.Weight.min(),X.Weight.max(),100)
varr= np.linspace(X.Volume.min(),X.Volume.max(),100)

cwarr=warr.reshape(-1,1)
cvarr=varr.reshape(-1,1)

arr=np.concatenate((cwarr,cvarr),axis=1)
pred=regr.predict(arr)

fig=plt.figure()
ax=fig.add_subplot(projection='3d')
ax.scatter(X.Weight,X.Volume,y)

X,Y,Z=axes3d.get_test_data(0.05)


plt.show()