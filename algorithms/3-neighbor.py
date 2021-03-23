import pandas
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import scale
from typing import Tuple

columns = [
    "Class",
    "Alcohol",
    "Malic acid",
    "Ash",
    "Alcalinity of ash",
    "Magnesium",
    "Total phenols",
    "Flavanoids",
    "Nonflavanoid phenols",
    "Proanthocyanins",
    "Color intensity",
    "Hue",
    "OD280/OD315 of diluted wines",
    "Proline",
]

data = pandas.read_csv("wine.data", index_col=False, names=columns)

x = data.loc[:, data.columns != "Class"]
y = data['Class']

cv = KFold(n_splits=5, shuffle=True, random_state=42)


def get_best_score(x: pandas.DataFrame, y: pandas.Series, cv) -> Tuple[float, int]:
    best_score, best_k = None, None

    for k in range(1, 51):
        model = KNeighborsClassifier(n_neighbors=k)
        score = cross_val_score(model, x, y, cv=cv, scoring="accuracy").mean()

        if best_score is None or score > best_score:
            best_score, best_k = score, k

    return best_score, best_k

bestscore, bestk = get_best_score(x, y, cv)
bestscore1, bestk1 = get_best_score(scale(x), y, cv)
print(bestk1)
print(bestscore1)