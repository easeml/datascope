import numpy as np
import pytest

from datascope.importance.common import SklearnModelAccuracy
from datascope.importance.shapley import ShapleyImportance, ImportanceMethod
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


@pytest.mark.parametrize(
    "method", [ImportanceMethod.BRUTEFORCE, ImportanceMethod.MONTECARLO, ImportanceMethod.NEIGHBOR]
)
def test_simple_1(method: ImportanceMethod):
    X = np.array([[0.9], [0], [1]], dtype=float)
    y = np.array([1, 0, 0], dtype=float)
    X_test = np.array([[1], [0]], dtype=float)
    y_test = np.array([1, 0], dtype=float)
    utility = SklearnModelAccuracy(KNeighborsClassifier(n_neighbors=1))

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
    utility = SklearnModelAccuracy(KNeighborsClassifier(n_neighbors=1))

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
    utility = SklearnModelAccuracy(LogisticRegression())

    importance = ShapleyImportance(method=method, utility=utility)
    importance.fit(X, y, provenance=provenance)
    scores = importance.score(X_test, y_test)
    result = np.argsort(list(scores))
    expected = [np.array([1, 0, 2], dtype=int), np.array([1, 2, 0], dtype=int)]
    assert any(np.array_equal(result, candidate) for candidate in expected)


ERROR_MARGIN = 5e-2


@pytest.mark.parametrize("trainsize", [3, 12])
@pytest.mark.parametrize("testsize", [3, 10])
@pytest.mark.parametrize("method", [ImportanceMethod.NEIGHBOR, ImportanceMethod.MONTECARLO])
def test_comparative_1(trainsize: int, testsize: int, method: ImportanceMethod):

    X, y = make_classification(
        n_samples=trainsize + testsize,
        n_features=1,
        n_redundant=0,
        n_informative=1,
        n_repeated=0,
        n_classes=2,
        n_clusters_per_class=1,
        random_state=7,
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=trainsize, test_size=testsize, random_state=7)

    utility = SklearnModelAccuracy(KNeighborsClassifier(n_neighbors=1))

    importance = ShapleyImportance(method=method, utility=utility, mc_truncation_steps=0)
    importance_bf = ShapleyImportance(method=ImportanceMethod.BRUTEFORCE, utility=utility)

    importances = importance.fit(X_train, y_train).score(X_test, y_test)
    importances_bf = importance_bf.fit(X_train, y_train).score(X_test, y_test)
    for i, i_bf in zip(importances, importances_bf):
        assert np.abs(i - i_bf) < ERROR_MARGIN


# @pytest.mark.parametrize("n_samples_train", [100, 500, 1000, 5000, 10000])
@pytest.mark.parametrize("n_samples_train", [500, 1000, 5000, 10000])
# @pytest.mark.parametrize("n_samples_test", [10, 50, 100])
@pytest.mark.parametrize("n_samples_test", [1000])
def test_neighbor_benchmark_1(n_samples_train: int, n_samples_test: int, benchmark):
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
    benchmark(importance.score, X_test, y_test)


if __name__ == "__main__":
    n_samples_train = 50000
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
