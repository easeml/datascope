import numpy as np


from datascope.importance.common import get_indices, one_hot_encode, pad_jagged_array


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
    query = np.array([1, 1, 0])
    result = get_indices(provenance, query)
    expected = np.array([1, 1, 0, 1])
    assert np.array_equal(result, expected)


def test_get_indices_2d_1():
    provenance = np.array(
        [[[1, 1, 0], [1, 0, 0]], [[1, 0, 0], [0, 0, 1]], [[0, 0, 1], [0, 1, 0]], [[0, 1, 0], [1, 0, 0]]]
    )
    query = np.array([1, 0, 0])
    result = get_indices(provenance, query)
    expected = np.array([1, 1, 0, 1])
    assert np.array_equal(result, expected)
