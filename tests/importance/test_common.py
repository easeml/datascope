import itertools
import numpy as np
from sklearn.neighbors import KNeighborsClassifier


from datascope.importance.utility import (
    SklearnModelEqualizedOddsDifference,
    SklearnModelRocAuc,
    compute_groupings,
    equalized_odds_diff,
)


def test_compute_groupings_1():
    n_test = 10
    n_features = 3
    n_groups = 2
    sensitive_feature = [1]
    expected = np.random.randint(0, n_groups, size=n_test)
    X_test = np.random.random((n_test, n_features))
    X_test[:, sensitive_feature] = expected[:, np.newaxis] + 10
    result = compute_groupings(X=X_test, sensitive_features=sensitive_feature)
    assert np.array_equal(result, expected)


def test_equalized_odds_diff_1():
    groupings = np.array([0, 0, 1, 1, 1, 1])
    y_test = np.array([1, 0, 1, 1, 1, 1])
    y_pred = np.array([1, 1, 1, 1, 0, 1])
    result = equalized_odds_diff(y_test, y_pred, groupings=groupings)
    expected = 1.0
    assert result == expected


def test_eod_elementwise_score_1():
    X = np.array(
        [
            [0.9, 0],
            [0, 0],
            [1, 1],
            [0.1, 1],
        ],
        dtype=float,
    )
    y = np.array([1, 0, 0, 1], dtype=int)
    X_test = np.array(
        [
            [1, 0],
            [0.1, 0],
            [0.9, 1],
            [0, 1],
        ],
        dtype=float,
    )
    y_test = np.array([1, 0, 1, 0], dtype=int)
    utility = SklearnModelEqualizedOddsDifference(KNeighborsClassifier(n_neighbors=1), sensitive_features=1)

    result = utility.elementwise_score(X, y, X_test, y_test)
    expected = np.array(
        [
            [0, 0, 0, 0],
            [0, -1, 0, 1],
        ]
    )
    assert np.array_equal(result, expected)


def test_auc_elementwise_score_1():
    X = np.zeros_like((4, 2))
    y = np.array([1, 0, 0, 1], dtype=int)
    X_test = np.zeros_like((3, 2))
    y_test = np.array([0, 0, 1], dtype=int)

    utility = SklearnModelRocAuc(KNeighborsClassifier(n_neighbors=1))
    result = utility.elementwise_score(X, y, X_test, y_test)
    for idx in itertools.product(range(len(y)), repeat=len(y_test)):
        idx = list(idx)
        y_pred = y[idx]
        expected_score = utility.metric(y_test, y_pred)
        obtained_score = result[y_pred, range(len(y_test))].sum(dtype=float)
        assert obtained_score == expected_score


def test_auc_elementwise_null_score_1():
    X = np.zeros_like((4, 2))
    y = np.array([1, 0, 0, 1], dtype=int)
    X_test = np.zeros_like((3, 2))
    y_test = np.array([0, 0, 1], dtype=int)

    utility = SklearnModelRocAuc(KNeighborsClassifier(n_neighbors=1))
    result = utility.elementwise_null_score(X, y, X_test, y_test)
    expected = np.array([0.0, 0.0, 0.5])
    assert np.array_equal(result, expected)
