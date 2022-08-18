import numpy as np
import pytest

from datascope.importance.common import SklearnModelAccuracy
from datascope.importance.shapley import ShapleyImportance, ImportanceMethod
from datascope.importance.banzhaf import BanzhafImportance
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

@pytest.mark.parametrize("method", [ImportanceMethod.BRUTEFORCE, ImportanceMethod.MONTECARLO])
def test_simple(method: ImportanceMethod):
    X = np.array([[0.9], [0], [1]], dtype=float)
    y = np.array([1, 0, 0], dtype=float)
    provenance = np.array([[0], [0, 2], [1, 2]], dtype=object)
    X_test = np.array([[1], [0]], dtype=float)
    y_test = np.array([1, 0], dtype=float)
    utility = SklearnModelAccuracy(LogisticRegression())

    importance = BanzhafImportance(method=method, utility=utility)
    importance.fit(X, y, provenance=provenance)
    scores = importance.score(X_test, y_test)
    result = np.argsort(list(scores))
    expected = [np.array([1, 0, 2], dtype=int), np.array([1, 2, 0], dtype=int)]
    assert any(np.array_equal(result, candidate) for candidate in expected)


ERROR_MARGIN = 5e-2


        
if __name__ == "__main__":
    n_samples_train = 5000
    n_samples_test = 1000

    X, y = make_classification(
        n_samples=n_samples_train + n_samples_test,
        n_features=1,
        n_redundant=0,
        n_informative=1,
        n_repeated=0,
        n_classes=2,
        n_clusters_per_class=1,
        random_state=7,
    )

    X, X_test, y, y_test = train_test_split(X, y, train_size=n_samples_train, test_size=n_samples_test, random_state=7)

    utility = SklearnModelAccuracy(KNeighborsClassifier(n_neighbors=1))
    importance = ShapleyImportance(method=ImportanceMethod.NEIGHBOR, utility=utility)
    importance.fit(X, y)
    importance.score(X_test, y_test)
