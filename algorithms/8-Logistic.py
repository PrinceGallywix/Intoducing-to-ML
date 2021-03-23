import pandas as pd
from sklearn.linear_model import SGDClassifier
from typing import Tuple
import numpy as np
from sklearn.metrics import roc_auc_score

data = pd.read_csv(r'C:\Users\Dmitriy\PycharmProjects\data-logistic.csv', header = None)

X = data.loc[:, 1:]
y = data[0]

def calc_w1(X: pd.DataFrame, y: pd.Series, w1: float, w2: float, k: float, C: float) -> float:
    l = len(y)
    S = 0
    for i in range(0, l):
        S += y[i] * X[1][i] * (1.0 - 1.0 / (1.0 + np.exp(-y[i] * (w1*X[1][i] + w2*X[2][i]))))

    return w1 + (k * (1.0 / l) * S) - k * C * w1

def calc_w2(X: pd.DataFrame, y: pd.Series, w1: float, w2: float, k: float, C: float) -> float:
    l = len(y)
    S = 0
    for i in range(0, l):
        S += y[i] * X[2][i] * (1.0 - 1.0 / (1.0 + np.exp(-y[i] * (w1*X[1][i] + w2*X[2][i]))))

    return w2 + (k * (1.0 / l) * S) - k * C * w2

def gradient_descent(X: pd.DataFrame, y: pd.Series, w1: float=0.0, w2: float=0.0,
         k: float=0.1, C: float=0.0, precision: float=1e-5, max_iter: int=10000) -> Tuple[float, float]:
    for i in range(max_iter):
        w1_prev, w2_prev = w1, w2
        w1, w2 = calc_w1(X, y, w1, w2, k, C), calc_w2(X, y, w1, w2, k, C)
        if np.sqrt((w1_prev - w1) ** 2 + (w2_prev - w2) ** 2) <= precision:
            break

    return w1, w2

w1, w2 = gradient_descent(X, y)
w1_reg, w2_reg = gradient_descent(X, y, C=10.0)

def a(X: pd.DataFrame, w1: float, w2: float) -> pd.Series:
    return 1.0 / (1.0 + np.exp(-w1 * X[1] - w2 * X[2]))

y_proba = a(X, w1, w2)
y_proba_reg = a(X, w1_reg, w2_reg)

auc = roc_auc_score(y, y_proba)
auc_reg = roc_auc_score(y, y_proba_reg)

print(f"{auc:.3f} {auc_reg:.3f}")