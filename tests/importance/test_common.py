import numpy as np
from sklearn.neighbors import KNeighborsClassifier


from datascope.importance.common import (
    SklearnModelEqualizedOddsDifference,
    compute_groupings,
    equalized_odds_diff,
    get_indices,
    one_hot_encode,
    pad_jagged_array,
    reshape,
)


def test_pad_jagged_array_2d_1():
    array = np.array([[1, 1], [2, 2, 2], [3]], dtype=object)
    result = pad_jagged_array(array, fill_value=-1)
    expected = np.array([[1, 1, -1], [2, 2, 2], [3, -1, -1]], dtype=object)
    assert np.array_equal(result, expected)


def test_pad_jagged_array_3d_1():
    array = np.array([[[1], [1, 1]], [[2, 2, 2], [2], [2, 2]], [[3]]], dtype=object)
    result = pad_jagged_array(array, fill_value=-1)
    expected = np.array(
        [
            [[1, -1, -1], [1, 1, -1], [-1, -1, -1]],
            [[2, 2, 2], [2, -1, -1], [2, 2, -1]],
            [[3, -1, -1], [-1, -1, -1], [-1, -1, -1]],
        ],
        dtype=object,
    )
    assert np.array_equal(result, expected)


def test_one_hot_encode_1d_1():
    array = np.array([1, 0, 2, 1])
    result = one_hot_encode(array)
    expected = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0]])
    assert np.array_equal(result, expected)


def test_one_hot_encode_2d_1():
    array = np.array([[1, 2], [0, 1], [2, 1]])
    result = one_hot_encode(array)
    expected = np.array([[[0, 1, 0], [0, 0, 1]], [[1, 0, 0], [0, 1, 0]], [[0, 0, 1], [0, 1, 0]]])
    assert np.array_equal(result, expected)


def test_one_hot_encode_merged_1d_1():
    array = np.array([1, 0, 2, 1])
    result = one_hot_encode(array, mergelast=True)
    expected = np.array([1, 1, 1])
    assert np.array_equal(result, expected)


def test_one_hot_encode_merged_2d_1():
    array = np.array([[1, 2], [0, 1], [2, 1]])
    result = one_hot_encode(array, mergelast=True)
    expected = np.array([[0, 1, 1], [1, 1, 0], [0, 1, 1]])
    assert np.array_equal(result, expected)


def test_get_indices_1d_1():
    provenance = np.array([[1, 1, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0]])
    provenance = np.reshape(provenance, (-1, 1, 3, 1))
    provenance = np.concatenate([provenance, np.zeros_like(provenance)], axis=3)
    query = np.array([1, 1, 0])
    result = get_indices(provenance, query)
    expected = np.array([1, 1, 0, 1])
    assert np.array_equal(result, expected)


def test_get_indices_2d_1():
    provenance = np.array(
        [[[1, 1, 0], [1, 0, 0]], [[1, 0, 0], [0, 0, 1]], [[0, 0, 1], [0, 1, 0]], [[0, 1, 0], [1, 0, 0]]]
    )
    provenance = np.reshape(provenance, (-1, 2, 3, 1))
    provenance = np.concatenate([provenance, np.zeros_like(provenance)], axis=3)
    provenance = reshape(provenance)
    query = np.array([1, 0, 0])
    result = get_indices(provenance, query)
    expected = np.array([1, 1, 0, 1])
    assert np.array_equal(result, expected)


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
    y = np.array([1, 0, 0, 1], dtype=float)
    X_test = np.array(
        [
            [1, 0],
            [0.1, 0],
            [0.9, 1],
            [0, 1],
        ],
        dtype=float,
    )
    y_test = np.array([1, 0, 1, 0], dtype=float)
    utility = SklearnModelEqualizedOddsDifference(KNeighborsClassifier(n_neighbors=1), sensitive_features=1)

    result = utility.elementwise_score(X, y, X_test, y_test)
    expected = np.array(
        [
            [0, -1, 0, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, -1, 0, 1],
        ]
    )
    assert np.array_equal(result, expected)
