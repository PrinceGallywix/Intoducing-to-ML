from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd

newsgroups = datasets.fetch_20newsgroups(
                    subset='all',
                    categories=['alt.atheism', 'sci.space']
             )
X = newsgroups.data
y = newsgroups.target

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

grid = {'C': np.power(10.0, np.arange(-5, 6))}
cv = KFold(n_splits=5, shuffle=True, random_state=241)
clf = SVC(kernel='linear', random_state=241)
gs = GridSearchCV(clf, grid, scoring='accuracy', cv=cv)
gs.fit(X, y)

C = gs.best_params_.get('C')


model = SVC(C=C, kernel='linear', random_state=241)
model.fit(X, y)


words = np.array(vectorizer.get_feature_names())
word_weights = pd.Series(model.coef_.data, index=words[model.coef_.indices], name="weight")
word_weights.index.name = "word"

top_words = word_weights.abs().sort_values(ascending=False).head(10)
print(top_words)

print(" ".join(top_words.index.sort_values(ascending=True)))