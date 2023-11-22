import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

df = pd.read_csv("titanic.csv")

df.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Embarked'], axis='columns', inplace=True)

inputs = df.drop('Survived', axis='columns')
target = df.Survived

# inputs.Sex = inputs.Sex.map({'male': 1, 'female': 2})

dummies = pd.get_dummies(inputs.Sex)

inputs = pd.concat([inputs, dummies], axis='columns')
inputs.drop(['Sex', 'female'], axis='columns', inplace=True)

# Muestra por pantalla las columnas en las que hay valores nulos
# print(inputs.columns[inputs.isna().any()])

# Rellena los valores nulos de Age
inputs.Age = inputs.Age.fillna(inputs.Age.mean())

X_train, X_test, y_train, y_test = train_test_split(inputs,target,test_size=0.3)
model = GaussianNB()
model.fit(X_train,y_train)
print(model.score(X_test,y_test))
print(model.predict(X_test[0:1]))
