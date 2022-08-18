import numpy as np
import pytest

from datascope.importance.common import SklearnModelAccuracy
from datascope.importance.shapley import ShapleyImportance, ImportanceMethod
from datascope.importance.banzhaf import BanzhafImportance
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

X = np.array([[0.9], [0], [1]], dtype=float)
y = np.array([1, 0, 1], dtype=float)
# provenance = np.array([[0], [0, 2], [1, 2]], dtype=object)
X_test = np.array([[1], [2]], dtype=float)
y_test = np.array([1, 3], dtype=float)
utility = SklearnModelAccuracy(LogisticRegression())

methods = [ImportanceMethod.BRUTEFORCE, ImportanceMethod.MONTECARLO]

list_of_scores = []
for method in methods:
    importance = ShapleyImportance(method=method, utility=utility)
    importance.fit(X, y, provenance=None)
    scores = importance.score(X_test, y_test)
    list_of_scores.append(scores)

print(list_of_scores)