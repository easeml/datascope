import numpy as np
import time
import warnings

from enum import Enum
from itertools import product
from scipy.special import comb

# from numba import prange, jit
from logging import Logger, getLogger
from numpy import ndarray
from scipy.sparse import csr_matrix, issparse, spmatrix
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import DistanceMetric
from sklearn.pipeline import Pipeline

from typing_extensions import Literal
from typing import Dict, List, Optional, Iterable, Set, Tuple

# from .add import ADD
from .common import DEFAULT_SEED, DistanceCallable, Utility, binarize, get_indices, reshape
from .importance import Importance


from .shapley_cy import compute_all_importances_cy

prange = range


class ImportanceMethod(str, Enum):
    BRUTEFORCE = "bruteforce"
    MONTECARLO = "montecarlo"
    NEIGHBOR = "neighbor"


DEFAULT_MC_ITERATIONS = 500
DEFAULT_MC_TIMEOUT = 0
DEFAULT_MC_TOLERANCE = 0.1
DEFAULT_MC_TRUNCATION_STEPS = 5
DEFAULT_NN_K = 1
DEFAULT_NN_DISTANCE = DistanceMetric.get_metric("minkowski").pairwise


def checknan(x: ndarray) -> bool:
    if (x.ndim == 0) and np.all(np.isnan(x)):
        return True
    else:
        return False


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


# def compile_add(sigma: ndarray, provenance: ndarray, units: ndarray, btuple: int) -> ADD:
#     pass


# def compute_shapley_add(
#     distances: ndarray,
#     utilities: ndarray,
#     provenance: ndarray,
#     units: ndarray,
# ) -> Iterable[float]:
#     pass


# @jit(nopython=True)
def get_unit_distances_and_utilities(
    distances: ndarray,
    utilities: ndarray,
    provenance: ndarray,
    units: ndarray,
    world: ndarray,
    simple_provenance: bool = False,
) -> Tuple[ndarray, ndarray]:

    n_test = distances.shape[1]
    n_units = len(units)
    # We check if we are dealing with the trivial situation, when we need only to return the trivial answer.
    if simple_provenance or (
        provenance.ndim == 3
        and provenance.shape[-1] == 1
        and provenance.shape[0] == provenance.shape[1]
        and np.all(np.equal(provenance[..., 0], np.eye(n_units)))
    ):
        return distances, utilities

    # Compute the minimal distance for each unit and each test example.
    unit_provenance = provenance[..., units, world].astype(np.bool_)
    unit_distances = np.zeros((n_units, n_test))  # TODO: Make this faster.
    unit_utilities = np.zeros((n_units, n_test))  # TODO: Make this faster.
    for i in range(n_units):
        # # Find minimal distance indices for each test example, among tuples associated with i-th unit.
        # idx = np.argmin(distances[unit_provenance[:, i]], axis=0)
        # # Given those indices, select the minimal distance value and the associated utility value.
        # unit_distances[i] = distances[unit_provenance[:, i]][idx, np.arange(n_test)]
        # unit_utilities[i] = utilities[unit_provenance[:, i]][idx, np.arange(n_test)]
        pidx = unit_provenance[:, i]
        pdistances = distances[pidx]
        putilities = utilities[pidx]
        for j in range(n_test):
            idx = np.argmin(distances[pidx, j])
            unit_distances[i, j] = pdistances[idx, j]
            unit_utilities[i, j] = putilities[idx, j]
    return unit_distances, unit_utilities


# @jit(nopython=True, nogil=True, cache=True)
def compute_all_importances(
    unit_distances: ndarray,
    unit_utilities: ndarray,
    null_scores: ndarray,
) -> ndarray:
    # Compute unit importances.
    n_units, n_test = unit_distances.shape
    all_importances = np.zeros((n_units + 1))
    unit_utilities = np.vstack((unit_utilities, null_scores))
    for j in prange(n_test):
        idxs = np.append(np.argsort(unit_distances[:, j]), [n_units])
        current = 0.0
        for i in prange(n_units - 1, -1, -1):
            i_1 = idxs[i]
            i_2 = idxs[i + 1]
            current += (unit_utilities[i_1, j] - unit_utilities[i_2, j]) / float(i + 1)
            all_importances[i_1] += current
        # all_importances /= n_units
    result = all_importances[:-1] / n_test
    return result


# @jit(nopython=True)
# @njit(parallel=False)
# @njit(parallel=True, fastmath=True)
def compute_shapley_1nn_mapfork(
    distances: ndarray,
    utilities: ndarray,
    provenance: ndarray,
    units: ndarray,
    world: ndarray,
    null_scores: Optional[ndarray] = None,
    simple_provenance: bool = False,
) -> Iterable[float]:

    # if not np.all(provenance.sum(axis=2) == 1):
    #     raise ValueError("The provenance of all data examples must reference at most one unit.")
    if not checknan(provenance):  # TODO: Remove this hack.
        provenance = np.squeeze(provenance, axis=1)

    # n_test = distances.shape[1]
    # n_units = len(units)

    # # Compute the minimal distance for each unit and each test example.
    # unit_provenance = provenance[..., units, world].astype(np.bool_)
    # unit_distances = np.zeros((n_units, n_test))  # TODO: Make this faster.
    # unit_utilities = np.zeros((n_units, n_test))  # TODO: Make this faster.
    # for i in prange(n_units):
    #     # # Find minimal distance indices for each test example, among tuples associated with i-th unit.
    #     # idx = np.argmin(distances[unit_provenance[:, i]], axis=0)
    #     # # Given those indices, select the minimal distance value and the associated utility value.
    #     # unit_distances[i] = distances[unit_provenance[:, i]][idx, np.arange(n_test)]
    #     # unit_utilities[i] = utilities[unit_provenance[:, i]][idx, np.arange(n_test)]
    #     for j in prange(n_test):
    #         idx = np.argmin(distances[unit_provenance[:, i], j])
    #         unit_distances[i, j] = distances[unit_provenance[:, i]][idx, j]
    #         unit_utilities[i, j] = utilities[unit_provenance[:, i]][idx, j]

    unit_distances, unit_utilities = distances, utilities
    if not checknan(provenance):  # TODO: Remove this hack.
        unit_distances, unit_utilities = get_unit_distances_and_utilities(
            distances, utilities, provenance, units, world, simple_provenance=simple_provenance
        )
    # unit_distances, unit_utilities = distances, utilities

    # # Compute unit importances.
    # all_importances = np.zeros((n_units + 1, n_test))
    # unit_utilities = np.vstack((unit_utilities, np.ones((1, n_test)) * 0.5))
    # for j in prange(n_test):
    #     idxs = np.append(np.argsort(unit_distances[:, j]), [n_units])
    #     for i in range(n_units - 1, -1, -1):
    #         all_importances[idxs[i], j] = all_importances[idxs[i + 1], j] + (
    #             unit_utilities[idxs[i], j] - unit_utilities[idxs[i + 1], j]
    #         ) / float(i + 1)
    n_test = distances.shape[1]
    null_scores = null_scores if null_scores is not None else np.zeros((1, n_test))
    all_importances = compute_all_importances_cy(unit_distances, unit_utilities, null_scores)

    # Aggregate results.
    # importances = np.mean(all_importances[:-1], axis=1)
    # importances = np.zeros(n_units)
    # for i in range(n_units):
    #     importances[i] = np.mean(all_importances[i])
    return all_importances


class ShapleyImportance(Importance):
    def __init__(
        self,
        method: Literal[ImportanceMethod.BRUTEFORCE, ImportanceMethod.MONTECARLO, ImportanceMethod.NEIGHBOR],
        utility: Utility,
        pipeline: Optional[Pipeline] = None,
        mc_iterations: int = DEFAULT_MC_ITERATIONS,
        mc_timeout: int = DEFAULT_MC_TIMEOUT,
        mc_tolerance: float = DEFAULT_MC_TOLERANCE,
        mc_truncation_steps: int = DEFAULT_MC_TRUNCATION_STEPS,
        mc_preextract: bool = False,
        nn_k: int = DEFAULT_NN_K,
        nn_distance: DistanceCallable = DEFAULT_NN_DISTANCE,
        seed: int = DEFAULT_SEED,
        logger: Optional[Logger] = None,
    ) -> None:
        super().__init__()
        self.method = ImportanceMethod(method)
        self.utility = utility
        self.pipeline = pipeline
        self.mc_iterations = mc_iterations
        self.mc_timeout = mc_timeout
        self.mc_tolerance = mc_tolerance
        self.mc_truncation_steps = mc_truncation_steps
        self.mc_preextract = mc_preextract
        self.nn_k = nn_k
        self.nn_distance = nn_distance
        self.X: Optional[ndarray] = None
        self.y: Optional[ndarray] = None
        self.provenance: Optional[ndarray] = None
        self._simple_provenance = False
        self.randomstate = np.random.RandomState(seed)
        self.logger = logger if logger is not None else getLogger(__name__)

    def _fit(self, X: ndarray, y: ndarray, provenance: Optional[ndarray] = None) -> "ShapleyImportance":
        self.X = X
        self.y = y

        if provenance is None:
            provenance = np.arange(X.shape[0])
            self._simple_provenance = True
        if checknan(provenance):
            self._simple_provenance = True
        if not checknan(provenance):  # TODO: Remove this hack.
            self.provenance = reshape(provenance)
        else:
            self.provenance = provenance
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
                self.mc_timeout,
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

        if checknan(provenance):  # TODO: Remove this hack.
            units = np.arange(X.shape[0])
            world = np.zeros_like(units, dtype=int)
            provenance = np.arange(X.shape[0])
            provenance = reshape(provenance)

        # Convert provenance and units to bit-arrays.
        provenance = binarize(provenance)

        # Apply the feature extraction pipeline if it was provided.
        if self.pipeline is not None:
            self.pipeline.fit(X, y=y)
            X_test = self.pipeline.transform(X_test)

        # Compute null score.
        null_score = self.utility.null_score(X, y, X_test, y_test)

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
            score = null_score
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                try:
                    X_train = X[indices]
                    y_train = y[indices]
                    if self.pipeline is not None:
                        X_train = self.pipeline.fit_transform(X_train, y=y)
                    score = self.utility(X_train, y_train, X_test, y_test, null_score)
                except (ValueError, RuntimeWarning, UserWarning):
                    pass

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
        timeout: int = DEFAULT_MC_TIMEOUT,
        tolerance: float = DEFAULT_MC_TOLERANCE,
        truncation_steps: int = DEFAULT_MC_TRUNCATION_STEPS,
    ) -> Iterable[float]:

        if checknan(provenance):  # TODO: Remove this hack.
            units = np.arange(X.shape[0])
            world = np.zeros_like(units, dtype=int)
            provenance = np.arange(X.shape[0])
            provenance = reshape(provenance)

        # Convert provenance and units to bit-arrays.
        provenance = binarize(provenance)

        # Apply the feature extraction pipeline if it was provided.
        X_tr = X
        X_ts = X_test
        if self.pipeline is not None:
            X_tr = self.pipeline.fit_transform(X, y=y)
            X_ts = self.pipeline.transform(X_test)

        # Compute mean score.
        null_score = self.utility.null_score(X_tr, y, X_ts, y_test)
        mean_score = self.utility.mean_score(X_tr, y, X_ts, y_test)

        # If pre-extract was specified, run feature extraction once for the whole dataset.
        if self.mc_preextract:
            X = X_tr  # TODO: Handle provenance if the pipeline is not a map pipeline.
            X_test = X_ts

        # Run a given number of iterations.
        n_units_total = provenance.shape[2]
        n_candidates_total = provenance.shape[3]
        n_units = len(units)
        simple_provenance = self._simple_provenance or bool(
            provenance.shape[1] == 1
            and provenance.shape[3] == 1
            and provenance.shape[0] == provenance.shape[2]
            and np.all(np.equal(provenance[:, 0, :, 0], np.eye(n_units)))
        )
        all_importances = np.zeros((n_units, iterations))
        all_truncations = np.ones(iterations, dtype=int) * n_units
        start_time = time.time()
        for i in range(iterations):
            idxs = self.randomstate.permutation(n_units)
            importance = np.zeros(n_units)
            query = np.zeros((n_units_total, n_candidates_total))

            new_score = null_score
            truncation_counter = 0

            for j, idx in enumerate(idxs):
                old_score = new_score

                # Get indices of data points selected based on the iteration query.
                query[units[idx], world[idx]] = 1
                indices = get_indices(provenance, query, simple_provenance=simple_provenance)

                # Train the model and score it. If we fail at any step we get zero score.
                new_score = null_score
                with warnings.catch_warnings():
                    warnings.simplefilter("error")
                    try:
                        X_train = X[indices]
                        y_train = y[indices]
                        X_ts = X_test
                        if self.pipeline is not None and not self.mc_preextract:
                            X_train = self.pipeline.fit_transform(X_train, y=y)
                            X_ts = self.pipeline.transform(X_test)
                        new_score = self.utility(X_train, y_train, X_ts, y_test, null_score=null_score)
                    except (ValueError, RuntimeWarning, UserWarning):
                        pass

                importance[idx] = new_score - old_score

                if np.abs(new_score - mean_score) <= np.abs(tolerance * mean_score):
                    truncation_counter += 1
                    if truncation_steps > 0 and truncation_counter > truncation_steps:
                        all_truncations[i] = j + 1
                        break
                else:
                    truncation_counter = 0

            all_importances[:, i] = importance

            # Check if we have timed out.
            elapsed_time = time.time() - start_time
            if timeout > 0 and elapsed_time > timeout:
                all_importances = all_importances[:, :i]
                break

        scores = np.average(all_importances, axis=1)
        truncations = np.average(all_truncations)
        self.logger.debug("Truncation ratio: %s", str(truncations / n_units))
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
        if not checknan(provenance):  # TODO: Remove this hack.
            provenance = binarize(provenance)

        # Apply the feature extraction pipeline if it was provided.
        if self.pipeline is not None:
            X = self.pipeline.fit_transform(X, y=y)
            X_test = self.pipeline.transform(X_test)

        # Compute the distances between training and text data examples.
        if issparse(X):
            assert isinstance(X, spmatrix)
            X = X.todense()
        if issparse(X_test):
            assert isinstance(X_test, spmatrix)
            X_test = X_test.todense()
        distances = distance(X, X_test)

        # Compute the utilitiy values between training and test labels.
        utilities = self.utility.elementwise_score(X_train=X, y_train=y, X_test=X_test, y_test=y_test)

        # Compute null scores.
        null_scores = self.utility.elementwise_null_score(X, y, X_test, y_test)

        if k == 1:
            return compute_shapley_1nn_mapfork(
                distances,
                utilities,
                provenance,
                units,
                world,
                simple_provenance=self._simple_provenance,
                null_scores=null_scores,
            )
        else:
            raise ValueError("The value '%d' for the k-parameter is not possible.")
