import pandas
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

df = pandas.read_csv("DecisionTrees/data2.csv")


# seleccionamos las colummnas 'feature' y la 'target'
features = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']

X = df[features]
y = df['Outcome']

dtree = DecisionTreeClassifier()
dtree = dtree.fit(X, y)
plt.figure(figsize=(120,80))

print(dtree.predict([[40, 10, 6, 1,43,543,1,3]]))

tree.plot_tree(dtree, feature_names=features)

plt.show()

