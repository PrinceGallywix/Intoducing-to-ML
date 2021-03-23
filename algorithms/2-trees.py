import numpy as np
from sklearn.tree import DecisionTreeClassifier
import pandas

data = pandas.read_csv('titanic.csv', index_col='PassengerId')
x = data.loc[:, ["Pclass", "Fare", "Age", "Sex"]]
x['Sex'] = x['Sex'].map({"male" : 0 , "female":1})
y = data["Survived"]

x = x.dropna()
y = y[x.index]
clf = DecisionTreeClassifier(random_state=241)
clf.fit(x,y)
importances = clf.feature_importances_
print(importances)