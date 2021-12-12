import numpy as np
import pytest

from datascope.importance.common import SklearnModelUtility
from datascope.importance.shapley import ShapleyImportance, ImportanceMethod
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


@pytest.mark.parametrize(
    "method", [ImportanceMethod.BRUTEFORCE, ImportanceMethod.MONTECARLO, ImportanceMethod.NEIGHBOR]
)
def test_simple_1(method: ImportanceMethod):
    X = np.array([[0.9], [0], [1]], dtype=float)
    y = np.array([1, 0, 0], dtype=float)
    X_test = np.array([[1], [0]], dtype=float)
    y_test = np.array([1, 0], dtype=float)
    utility = SklearnModelUtility(LogisticRegression(), accuracy_score)

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
    utility = SklearnModelUtility(LogisticRegression(), accuracy_score)

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
