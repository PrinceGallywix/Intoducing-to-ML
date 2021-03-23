import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score

df = pd.read_csv('abalone.csv')

df['Sex'] = df['Sex'].map(lambda x: 1 if x == 'M' else (-1 if x == 'F' else 0))
X = df.loc[:, "Sex":"ShellWeight"]
y = df["Rings"]

cv = KFold(n_splits=5, shuffle=True, random_state=1)

scores = []
for i in range(1,51):
    clf = RandomForestRegressor(n_estimators=i, random_state=1, n_jobs=-1)
    score = cross_val_score(clf, X, y, cv=cv, scoring="r2").mean()
    scores.append(score)

for i, score in enumerate(scores):
    if score > 0.52:
        print(str(i + 1))
        break


pd.Series(scores).plot()
