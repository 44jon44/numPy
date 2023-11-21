import pandas
from sklearn import linear_model
import matplotlib.pyplot as plt
from scipy import stats

import numpy as np

df = pandas.read_csv("data.csv")

x = df[['Volume']]
y = df['CO2']

res = linear_model.LinearRegression()
res.fit(x, y)
warr = np

mymodel = np.poly1d(np.polyfit(x, y, 2))

myline = np.linspace(1, 22, 100)

plt.scatter(x, y)
plt.plot(myline, mymodel(myline))
plt.show()
