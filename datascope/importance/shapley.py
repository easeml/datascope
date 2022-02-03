import numpy as np

from enum import Enum
from itertools import product
from math import comb

# from numba import prange
from numpy import ndarray
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import DistanceMetric
from typing import Dict, List, Literal, Optional, Iterable, Set, Tuple

from .add import ADD
from .common import DEFAULT_SEED, DistanceCallable, Utility, binarize, get_indices, reshape
from .importance import Importance


prange = range


class ImportanceMethod(Enum):
    BRUTEFORCE = "bruteforce"
    MONTECARLO = "montecarlo"
    NEIGHBOR = "neighbor"


DEFAULT_MC_ITERATIONS = 500
DEFAULT_MC_TOLERANCE = 0.1
DEFAULT_MC_TRUNCATION_STEPS = 5
DEFAULT_NN_K = 1
DEFAULT_NN_DISTANCE = DistanceMetric.get_metric("minkowski").pairwise


def factorize_provenance(
    provenance: ndarray, units: ndarray
) -> Tuple[List[Set[int]], List[Dict[int, Dict[Tuple[int, ...], List[int]]]]]:
    assert provenance.ndim == 2

    # Compute a mapping between unique polynomials and a list of indices of tuples they are associated with.
    polynomial_tuples: Dict[Tuple[int, ...], List[int]] = dict()
    unit_polynomials: Dict[int, Set[Tuple[int, ...]]] = dict((u, set()) for u in units)
    sunits = set(units)
    for i in range(provenance.shape[0]):
        # We assume that all units not explicitly present in the units list are set to 1.
        p = tuple(x for x in sorted(provenance[i][provenance[i] != -1]) if x in sunits)
        polynomial_tuples.setdefault(p, []).append(i)
        for u in p:
            unit_polynomials[u].add(p)

    unit_occurences: Dict[int, int] = dict((u, 0) for u in units)  # Number of distinct polynomials each unit occurs in.
    # tuples: Dict[int, Dict[Tuple[int, ...], List[int]]] = dict((u, {}) for u in units)
    unit_coocurrences = np.zeros((len(units), len(units)), dtype=int)  # Unit co-occurence matrix.
    unit_indices = dict((int(u), i) for (i, u) in enumerate(units))  # Indices of units in the provided list.
    for p, idxs in polynomial_tuples.items():
        for j, u in enumerate(p):
            unit_occurences[u] += 1
            # tuples[u].setdefault(p, []).extend(idxs)
            for v in p[j + 1 :]:  # noqa: E203
                unit_coocurrences[unit_indices[u], unit_indices[v]] = 1

    # Obtain connected components.
    n_components, component_labels = connected_components(csr_matrix(unit_coocurrences), directed=False)

    # Use connected components to extract factors and leaves.
    factors: List[Set[int]] = [set() for _ in range(n_components)]
    leaves: List[Dict[int, Dict[Tuple[int, ...], List[int]]]] = [{} for _ in range(n_components)]
    for i, c in enumerate(component_labels):
        u = units[i]
        if unit_occurences[u] == 1:
            leaves[c][u] = dict((p, polynomial_tuples[p]) for p in unit_polynomials[u])
        elif unit_occurences[u] > 1:
            factors[c].add(u)

    return factors, leaves


def compile_add(sigma: ndarray, provenance: ndarray, units: ndarray, btuple: int) -> ADD:
    pass


def compute_shapley_add(
    distances: ndarray,
    utilities: ndarray,
    provenance: ndarray,
    units: ndarray,
) -> Iterable[float]:
    pass


# @jit(nopython=True)
# @njit(parallel=False)
# @njit(parallel=True, fastmath=True)
def compute_shapley_1nn_mapfork(
    distances: ndarray, utilities: ndarray, provenance: ndarray, units: ndarray, world: ndarray
) -> Iterable[float]:

    if not np.all(provenance.sum(axis=2) == 1):
        raise ValueError("The provenance of all data examples must reference at most one unit.")
    provenance = np.squeeze(provenance, axis=1)

    n_test = distances.shape[1]
    n_units = len(units)

    # Compute the minimal distance for each unit and each test example.
    unit_provenance = provenance[..., units, world].astype(np.bool_)
    unit_distances = np.zeros((n_units, n_test))  # TODO: Make this faster.
    unit_utilities = np.zeros((n_units, n_test))  # TODO: Make this faster.
    for i in prange(n_units):
        # # Find minimal distance indices for each test example, among tuples associated with i-th unit.
        # idx = np.argmin(distances[unit_provenance[:, i]], axis=0)
        # # Given those indices, select the minimal distance value and the associated utility value.
        # unit_distances[i] = distances[unit_provenance[:, i]][idx, np.arange(n_test)]
        # unit_utilities[i] = utilities[unit_provenance[:, i]][idx, np.arange(n_test)]
        for j in prange(n_test):
            idx = np.argmin(distances[unit_provenance[:, i], j])
            unit_distances[i, j] = distances[unit_provenance[:, i]][idx, j]
            unit_utilities[i, j] = utilities[unit_provenance[:, i]][idx, j]

    # Compute unit importances.
    all_importances = np.zeros((n_units + 1, n_test))
    unit_utilities = np.vstack((unit_utilities, np.ones((1, n_test)) * 0.5))
    for j in prange(n_test):
        idxs = np.append(np.argsort(unit_distances[:, j]), [n_units])
        for i in range(n_units - 1, -1, -1):
            all_importances[idxs[i], j] = all_importances[idxs[i + 1], j] + (
                unit_utilities[idxs[i], j] - unit_utilities[idxs[i + 1], j]
            ) / float(i + 1)

    # Aggregate results.
    # importances = np.mean(all_importances[:-1], axis=1)
    importances = np.zeros(n_units)
    for i in prange(n_units):
        importances[i] = np.mean(all_importances[i])
    return importances


class ShapleyImportance(Importance):
    def __init__(
        self,
        method: Literal[ImportanceMethod.BRUTEFORCE, ImportanceMethod.MONTECARLO, ImportanceMethod.NEIGHBOR],
        utility: Utility,
        mc_iterations: int = DEFAULT_MC_ITERATIONS,
        mc_tolerance: float = DEFAULT_MC_TOLERANCE,
        mc_truncation_steps: int = DEFAULT_MC_TRUNCATION_STEPS,
        nn_k: int = DEFAULT_NN_K,
        nn_distance: DistanceCallable = DEFAULT_NN_DISTANCE,
        seed: int = DEFAULT_SEED,
    ) -> None:
        super().__init__()
        self.method = ImportanceMethod(method)
        self.utility = utility
        self.mc_iterations = mc_iterations
        self.mc_tolerance = mc_tolerance
        self.mc_truncation_steps = mc_truncation_steps
        self.nn_k = nn_k
        self.nn_distance = nn_distance
        self.X: Optional[ndarray] = None
        self.y: Optional[ndarray] = None
        self.provenance: Optional[ndarray] = None
        self.randomstate = np.random.RandomState(seed)

    def _fit(self, X: ndarray, y: ndarray, provenance: Optional[ndarray] = None) -> "ShapleyImportance":
        self.X = X
        self.y = y
        if provenance is None:
            provenance = np.arange(len(X))
        self.provenance = reshape(provenance)
        return self

    def _score(self, X: ndarray, y: Optional[ndarray] = None, **kwargs) -> Iterable[float]:
        if self.X is None or self.y is None or self.provenance is None:
            raise ValueError("The fit function was not called first.")
        if y is None:
            raise ValueError("The 'y' argument cannot be None.")

        units = kwargs.get("units", np.unique(self.provenance))
        units = np.delete(units, np.where(units == -1))
        world = kwargs.get("world", np.zeros_like(units, dtype=int))
        return self._shapley(self.X, self.y, X, y, self.provenance, units, world)

    def _shapley(
        self,
        X: ndarray,
        y: ndarray,
        X_test: ndarray,
        y_test: ndarray,
        provenance: ndarray,
        units: ndarray,
        world: ndarray,
    ) -> Iterable[float]:
        if self.method == ImportanceMethod.BRUTEFORCE:
            return self._shapley_bruteforce(X, y, X_test, y_test, provenance, units, world)
        elif self.method == ImportanceMethod.MONTECARLO:
            return self._shapley_montecarlo(
                X,
                y,
                X_test,
                y_test,
                provenance,
                units,
                world,
                self.mc_iterations,
                self.mc_tolerance,
                self.mc_truncation_steps,
            )
        elif self.method == ImportanceMethod.NEIGHBOR:
            return self._shapley_neighbor(X, y, X_test, y_test, provenance, units, world, self.nn_k, self.nn_distance)
        else:
            raise ValueError("Unknown method '%s'." % self.method)

    def _shapley_bruteforce(
        self,
        X: ndarray,
        y: ndarray,
        X_test: ndarray,
        y_test: ndarray,
        provenance: ndarray,
        units: ndarray,
        world: ndarray,
    ) -> Iterable[float]:

        # Convert provenance and units to bit-arrays.
        provenance = binarize(provenance)

        # Compute null score.
        null_score = self.utility.null_score(X, y, X_test, y_test)
        null_score = 0.5

        # Iterate over all subsets of units.
        n_units_total = provenance.shape[2]
        n_candidates_total = provenance.shape[3]
        n_units = len(units)
        importance = np.zeros(n_units)
        for iteration in product(*[[0, 1] for _ in range(n_units)]):
            iter = np.array(iteration)
            s_iter = np.sum(iter)

            # Get indices of data points selected based on the iteration query.
            query = np.zeros((n_units_total, n_candidates_total))
            query[units, world] = iter
            indices = get_indices(provenance, query)

            # Train the model and score it. If we fail at any step we get zero score.
            X_train = X[indices]
            y_train = y[indices]
            score = self.utility(X_train, y_train, X_test, y_test, null_score)

            # Compute the factor and update the Shapley values of respective units.
            factor_0 = -1.0 / (comb(n_units - 1, min(s_iter, n_units - 1)) * n_units)
            factor_1 = 1.0 / (comb(n_units - 1, max(s_iter - 1, 0)) * n_units)
            importance += score * ((1 - iter) * factor_0 + iter * factor_1)

        return importance

    def _shapley_montecarlo(
        self,
        X: ndarray,
        y: ndarray,
        X_test: ndarray,
        y_test: ndarray,
        provenance: ndarray,
        units: ndarray,
        world: ndarray,
        iterations: int = DEFAULT_MC_ITERATIONS,
        tolerance: float = DEFAULT_MC_TOLERANCE,
        truncation_steps: int = DEFAULT_MC_TRUNCATION_STEPS,
    ) -> Iterable[float]:

        # Convert provenance and units to bit-arrays.
        provenance = binarize(provenance)

        # Compute mean score.
        null_score = self.utility.null_score(X, y, X_test, y_test)
        mean_score = self.utility.mean_score(X, y, X_test, y_test)

        # Run a given number of iterations.
        n_units_total = provenance.shape[2]
        n_candidates_total = provenance.shape[3]
        n_units = len(units)
        truncation_counter = 0
        all_importances = np.zeros((n_units, iterations))
        for i in range(iterations):
            idxs = self.randomstate.permutation(n_units)
            importance = np.zeros(n_units)
            query = np.zeros((n_units_total, n_candidates_total))

            new_score = null_score

            for idx in idxs:
                old_score = new_score

                # Get indices of data points selected based on the iteration query.
                query[units[idx], world[idx]] = 1
                indices = get_indices(provenance, query)

                # Train the model and score it. If we fail at any step we get zero score.
                X_train = X[indices]
                y_train = y[indices]
                new_score = self.utility(X_train, y_train, X_test, y_test, null_score=null_score)

                importance[idx] = (new_score - old_score) / n_units

                if np.abs(new_score - mean_score) <= tolerance * mean_score:
                    truncation_counter += 1
                    if truncation_counter > truncation_steps:
                        break
                else:
                    truncation_counter = 0

            all_importances[:, i] = importance

        scores = np.average(all_importances, axis=1)
        return scores

    def _shapley_neighbor(
        self,
        X: ndarray,
        y: ndarray,
        X_test: ndarray,
        y_test: ndarray,
        provenance: ndarray,
        units: ndarray,
        world: ndarray,
        k: int,
        distance: DistanceCallable,
    ) -> Iterable[float]:

        # Convert provenance and units to bit-arrays.
        provenance = binarize(provenance)

        # Compute the distances between training and text data examples.
        distances = distance(X, X_test)

        # Compute the utilitiy values between training and test labels.
        # TODO: Enable different element-wise utilities.
        utilities = np.equal.outer(y, y_test).astype(float)

        if k == 1:
            return compute_shapley_1nn_mapfork(distances, utilities, provenance, units, world)
        else:
            raise ValueError("The value '%d' for the k-parameter is not possible.")
