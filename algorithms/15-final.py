import datetime
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression


#                 1st method


train = pd.read_csv("features.csv", index_col="match_id")
train.drop([
    "duration",
    "tower_status_radiant",
    "tower_status_dire",
    "barracks_status_radiant",
    "barracks_status_dire",
], axis=1, inplace=True)

count_na = len(train) - train.count()
count_na[count_na > 0].sort_values(ascending=False) / len(train)


train.fillna(0, inplace=True)
X_train = train.drop("radiant_win", axis=1)
y_train = train["radiant_win"]

cv = KFold(n_splits=5, shuffle=True)


def score_gb(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    scores = {}

    for n_estimators in [10, 20, 30, 50, 100, 250]:
        print(f"n_estimators={n_estimators}")
        model = GradientBoostingClassifier(n_estimators=n_estimators)

        start_time = datetime.datetime.now()
        score = cross_val_score(model, X, y, cv=cv, scoring="roc_auc").mean()
        print(f"Score: {score:.3f}")
        print(f"Time: {datetime.datetime.now() - start_time}")
        scores[n_estimators] = score
        print()
    return pd.Series(scores)


scores = score_gb(X_train, y_train)
scores.plot()


#           2nd method

Scale = StandardScaler()
X_train = pd.DataFrame(Scale.fit_transform(X_train), index=X_train.index, columns=X_train.columns)


def score_lr(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    scores = {}
    for i in range(-5, 6):
        C = 10.0 ** i
        print(f"C={C}")
        model = LogisticRegression(C=C, random_state=42)
        start_time = datetime.datetime.now()
        score = cross_val_score(model, X, y, cv=cv, scoring="roc_auc").mean()
        print(f"Score: {score:.3f}")
        print(f"Time: {datetime.datetime.now() - start_time}")

        scores[i] = score
        print()

    return pd.Series(scores)


scores = score_lr(X_train, y_train)
scores.plot()


def print_best_lr_score(scores: pd.Series):
    best_iteration = scores.sort_values(ascending=False).head(1)
    best_C = 10.0 ** best_iteration.index[0]
    best_score = best_iteration.values[0]

    print(f"Наилучшее значение  AUC-ROC  при C = {best_C:.2f}  равняется {best_score:.2f}.")


print_best_lr_score(scores)
hero_columns = [f"r{i}_hero" for i in range (1, 6)] + [f"d{i}_hero" for i in range (1, 6)]
cat_columns = ["lobby_type"] + hero_columns
X_train.drop(cat_columns, axis=1, inplace=True)
scores = score_lr(X_train, y_train)
scores.plot()

print_best_lr_score(scores)
unique_heroes = np.unique(train[hero_columns].values.ravel())
N = max(unique_heroes)
print(f"Число уникальных героев в train: {len(unique_heroes)}. Максимальный ID героя: {N}.")


def get_pick(data: pd.DataFrame) -> pd.DataFrame:
    X_pick = np.zeros((data.shape[0], N))

    for i, match_id in enumerate(data.index):
        for p in range(1, 6):
            X_pick[i, data.loc[match_id, f"r{p}_hero"] - 1] = 1
            X_pick[i, data.loc[match_id, f"d{p}_hero"] - 1] = -1

    return pd.DataFrame(X_pick, index=data.index, columns=[f"hero_{i}" for i in range(N)])


X_pick = get_pick(train)
X_pick.head()


X_train = pd.concat([X_train, X_pick], axis=1)

scores = score_lr(X_train, y_train)
scores.plot()

print_best_lr_score(scores)
model = LogisticRegression(C=0.1)
model.fit(X_train, y_train)
test = pd.read_csv("features_test.csv", index_col="match_id")
test.fillna(0, inplace=True)

X_test = pd.DataFrame(Scale.transform(test), index=test.index, columns=test.columns)
X_test.drop(cat_columns, axis=1, inplace=True)
X_test = pd.concat([X_test, get_pick(test)], axis=1)
X_test.head()

predict_it = pd.Series(model.predict_proba(X_test)[:, 1])
predict_it.describe()

predict_it.plot.hist(bins=30)
