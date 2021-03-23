from sklearn.neighbors import KNeighborsRegressor
from sklearn.datasets import load_boston
import sklearn
from sklearn.model_selection import KFold, cross_val_score
import numpy as np
from typing import Tuple
from sklearn.preprocessing import scale

data = sklearn.datasets.load_boston()

x = data.data
y = data.target
x = scale(x)
cv = KFold(n_splits=5, shuffle=True, random_state=42)

def get_best_par(x:np.array, y:np.array, cv) -> Tuple[float, float]:
    bestscore, bestpar = None, None

    for k in np.linspace(1,10, 200):
        model = KNeighborsRegressor(p=k, n_neighbors=5, weights='distance')
        score =  cross_val_score(model, x, y, cv=cv, scoring='neg_mean_squared_error').mean()

        if bestscore is None or bestscore < score:
            bestscore = score
            bestpar = k

    return bestscore, bestpar

score, par = get_best_par(x, y, cv)

print(score, par)
