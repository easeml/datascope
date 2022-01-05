import numpy as np
import pytest

from datascope.importance.common import SklearnModelUtility
from datascope.importance.shapley import ShapleyImportance, ImportanceMethod
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


@pytest.mark.parametrize(
    "method", [ImportanceMethod.BRUTEFORCE, ImportanceMethod.MONTECARLO, ImportanceMethod.NEIGHBOR]
)
def test_simple_1(method: ImportanceMethod):
    X = np.array([[0.9], [0], [1]], dtype=float)
    y = np.array([1, 0, 0], dtype=float)
    X_test = np.array([[1], [0]], dtype=float)
    y_test = np.array([1, 0], dtype=float)
    utility = SklearnModelUtility(KNeighborsClassifier(n_neighbors=1), accuracy_score)

    importance = ShapleyImportance(method=method, utility=utility)
    importance.fit(X, y)
    scores = importance.score(X_test, y_test)
    result = np.argsort(list(scores))
    expected = np.array([2, 0, 1], dtype=int)
    assert np.array_equal(result, expected)


@pytest.mark.parametrize(
    "method", [ImportanceMethod.BRUTEFORCE, ImportanceMethod.MONTECARLO, ImportanceMethod.NEIGHBOR]
)
def test_simple_2(method: ImportanceMethod):
    X = np.array([[0.9], [0], [1], [0]], dtype=float)
    y = np.array([1, 0, 0, 1], dtype=float)
    provenance = np.array([0, 0, 1, 1], dtype=object)
    X_test = np.array([[1], [0]], dtype=float)
    y_test = np.array([1, 0], dtype=float)
    utility = SklearnModelUtility(KNeighborsClassifier(n_neighbors=1), accuracy_score)

    importance = ShapleyImportance(method=method, utility=utility)
    importance.fit(X, y, provenance=provenance)
    scores = importance.score(X_test, y_test)
    result = np.argsort(list(scores))
    expected = np.array([1, 0], dtype=int)
    assert np.array_equal(result, expected)


@pytest.mark.parametrize("method", [ImportanceMethod.BRUTEFORCE, ImportanceMethod.MONTECARLO])
def test_simple_3(method: ImportanceMethod):
    X = np.array([[0.9], [0], [1]], dtype=float)
    y = np.array([1, 0, 0], dtype=float)
    provenance = np.array([[0], [0, 2], [1, 2]], dtype=object)
    X_test = np.array([[1], [0]], dtype=float)
    y_test = np.array([1, 0], dtype=float)
    utility = SklearnModelUtility(LogisticRegression(), accuracy_score)

    importance = ShapleyImportance(method=method, utility=utility)
    importance.fit(X, y, provenance=provenance)
    scores = importance.score(X_test, y_test)
    result = np.argsort(list(scores))
    expected = [np.array([1, 0, 2], dtype=int), np.array([1, 2, 0], dtype=int)]
    assert any(np.array_equal(result, candidate) for candidate in expected)


@pytest.mark.parametrize("n_samples", [100, 500, 1000])
def test_neighbor_benchmark_1(n_samples: int, benchmark):

    X, y = make_classification(
        n_samples=n_samples,
        n_features=1,
        n_redundant=0,
        n_informative=1,
        n_repeated=0,
        n_classes=2,
        n_clusters_per_class=1,
        random_state=7,
    )
    X, X_test, y, y_test = train_test_split(X, y, test_size=0.1, random_state=7)

    utility = SklearnModelUtility(KNeighborsClassifier(n_neighbors=1), accuracy_score)
    importance = ShapleyImportance(method=ImportanceMethod.NEIGHBOR, utility=utility)
    importance.fit(X, y)
    benchmark(importance.score, X_test, y_test)
