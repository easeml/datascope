#!python
#cython: language_level=3

import numpy as np


cimport numpy as np
cimport cython
np.import_array()

FLOAT = np.float64
INT = np.int64
ctypedef np.float_t FLOAT_t
ctypedef np.int_t INT_t


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
@cython.profile(True)
def compute_all_importances_cy(np.ndarray[INT_t, ndim=2] unit_labels, np.ndarray[FLOAT_t, ndim=2] unit_distances, np.ndarray[FLOAT_t, ndim=2] label_utilities, np.ndarray[FLOAT_t, ndim=1] null_scores):

    assert unit_labels.dtype == INT and unit_distances.dtype == FLOAT and label_utilities.dtype == FLOAT

    # Compute unit importances.
    cdef int n_units = unit_distances.shape[0]
    cdef int n_units_p = n_units + 1
    cdef int n_test = unit_distances.shape[1]
    cdef int n_classes = label_utilities.shape[0]
    cdef np.ndarray[np.int_t, ndim=2] idxs
    cdef np.ndarray[FLOAT_t, ndim=1] all_importances
    cdef int i, j
    cdef int i_1, i_2
    cdef float current

    all_importances = np.zeros([n_units + 1], dtype=FLOAT)
    unit_labels = np.vstack((unit_labels, np.repeat(n_classes, n_test)))
    label_utilities = np.vstack((label_utilities, null_scores))
    idxs = np.vstack((np.argsort(unit_distances, axis=0), np.full((1, n_test), n_units, dtype=int)))

    for j in range(n_test):
        current = 0.0
        for i in range(n_units - 1, -1, -1):
            i_1 = idxs[i, j]
            i_2 = idxs[i + 1, j]
            l_1 = unit_labels[i_1, j]
            l_2 = unit_labels[i_2, j]
            u_1 = label_utilities[l_1, j]
            u_2 = label_utilities[l_2, j]
            current += (u_1 - u_2) / (i + 1)
            all_importances[i_1] += current
    return all_importances[:-1] / n_test
