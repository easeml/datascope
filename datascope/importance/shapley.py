import numpy as np
import time
import warnings

from enum import Enum
from itertools import chain, product
from scipy.special import comb

# from numba import prange, jit
from logging import Logger, getLogger
from numpy.typing import NDArray
from scipy.sparse import csr_matrix, issparse, spmatrix
from scipy.sparse.csgraph import connected_components
from sklearn.preprocessing import LabelEncoder

try:
    from sklearn.metrics import DistanceMetric
except ImportError:
    from sklearn.neighbors import DistanceMetric

from sklearn.pipeline import Pipeline

from typing_extensions import Literal
from typing import Dict, List, Optional, Iterable, Set, Tuple, Sequence, Hashable, Union

from ..utility import Provenance
from .common import DEFAULT_SEED, DistanceCallable, Utility
from .importance import Importance
from .oracle import ShapleyOracle, ATally


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

# This should ensure that we never need more than a few GB of memory for computing Shapley values.
BATCH_DISTANCE_MATRIX_SIZE = 1024 * 1024 * 32


def get_test_batch_size(n_train: int, n_test: int) -> int:
    matrix_size = n_train * n_test
    n_matrices = max(matrix_size // BATCH_DISTANCE_MATRIX_SIZE, 1)
    return max(n_test // n_matrices, n_test)


def checknan(x: NDArray) -> bool:
    if (x.ndim == 0) and np.all(np.isnan(x)):
        return True
    else:
        return False


def factorize_provenance(
    provenance: NDArray, units: NDArray
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


def argmin(a):
    return min(range(len(a)), key=lambda x: a[x])


# @jit(nopython=True)
def get_unit_labels_and_distances(
    labels: NDArray,
    distances: NDArray,
    provenance: Provenance,
    units: NDArray,
    world: NDArray,
) -> Tuple[NDArray, NDArray]:
    n_train, n_test, n_units = distances.shape[0], distances.shape[1], len(units)
    assert labels.ndim == 1
    assert len(labels) == n_train

    # If the provenance is simple (i.e. each tuple has a single distinct unit and they are sorted)
    # then we just apply the simple case (the label vector needs to be broadcast only).
    if provenance.is_simple:
        labels = np.broadcast_to(np.expand_dims(labels, axis=1), (n_train, n_test))
        return labels, distances

    distances = np.array(distances, dtype=float)
    # Compute the minimal distance for each unit and each test example.
    query = np.zeros((provenance.num_units,), dtype=int)
    unit_labels = np.zeros((n_units, n_test), dtype=int)
    unit_distances = np.zeros((n_units, n_test))  # TODO: Make this faster.
    # for i, unit in enumerate(units):
    for i in range(n_units):
        # Find minimal distance indices for each test example, among tuples associated with i-th unit.
        unit = units[i]
        query[unit] = world[i]
        gidx = provenance.query(query)
        glabels = labels[gidx]
        gdistances = np.array(distances[gidx], dtype=float)
        gidx_min = np.argmin(gdistances, axis=0)
        for j in range(n_test):
            idx = gidx_min[j]
            unit_labels[i, j] = glabels[idx]
            unit_distances[i, j] = gdistances[idx, j]
        query[unit] = 0
    return unit_labels, unit_distances


# @jit(nopython=True, nogil=True, cache=True)
def compute_all_importances(
    unit_labels: NDArray,
    unit_distances: NDArray,
    label_utilities: NDArray,
    null_scores: NDArray,
) -> NDArray:
    # Compute unit importances.
    n_units, n_test = unit_distances.shape
    all_importances = np.zeros((n_units + 1))
    unit_labels = np.vstack((unit_labels, np.repeat(label_utilities.shape[0], n_test)))
    label_utilities = np.vstack((label_utilities, null_scores))
    for j in prange(n_test):
        idxs = np.append(np.argsort(unit_distances[:, j]), [n_units])
        current = 0.0
        for i in prange(n_units - 1, -1, -1):
            i_1 = idxs[i]
            i_2 = idxs[i + 1]
            current += (label_utilities[unit_labels[i_1, j], j] - label_utilities[unit_labels[i_2, j], j]) / float(
                i + 1
            )
            all_importances[i_1] += current
        # all_importances /= n_units
    result = all_importances[:-1] / n_test
    return result


# @jit(nopython=True)
# @njit(parallel=False)
# @njit(parallel=True, fastmath=True)
def compute_shapley_1nn_mapfork(
    labels: NDArray,
    distances: NDArray,
    label_utilities: NDArray,
    provenance: Provenance,
    units: NDArray,
    world: NDArray,
    null_scores: Optional[NDArray] = None,
) -> NDArray:
    # Compute the minimal distance for each unit and each test example.
    unit_labels, unit_distances = get_unit_labels_and_distances(labels, distances, provenance, units, world)

    # Compute unit importances.
    n_test = distances.shape[1]
    null_scores = null_scores if null_scores is not None else np.zeros((1, n_test))
    all_importances = compute_all_importances_cy(unit_labels, unit_distances, label_utilities, null_scores)

    return all_importances


def compute_shapley_add(
    labels: NDArray,
    distances: NDArray,
    label_utilities: NDArray,
    provenance: Provenance,
    units: NDArray,
    world: NDArray,
    max_cardinality: Optional[int] = None,
    num_neighbors: int = 1,
    num_classes: int = 2,
    null_scores: Optional[NDArray] = None,
) -> NDArray:
    if max_cardinality is None or max_cardinality >= distances.shape[0]:
        max_cardinality = distances.shape[0] - 1
    n_units, n_tuples, n_test = len(units), distances.shape[0], distances.shape[1]
    all_importances = np.zeros((n_units), dtype=float)
    null_scores = null_scores if null_scores is not None else np.zeros((1, n_test))
    atype = ATally[max_cardinality, num_neighbors, num_classes]  # type: ignore

    for j in prange(n_test):
        oracle = ShapleyOracle(provenance=provenance, labels=labels, distances=distances[:, j], atype=atype)
        for t1, t2 in product(range(n_tuples), chain(range(n_tuples), [None])):
            for i, unit in enumerate(units):
                result = oracle.query(target=provenance.units[unit], boundary_with=t1, boundary_without=t2)
                assert result is not None
                for avalue, rvalue in result.items():
                    if (
                        avalue.is_inf
                        or avalue.tupletally == 0
                        or rvalue <= 0
                        or (np.sum(avalue.labeltally_with) != num_neighbors and t1 is not None)  # type: ignore
                        or (np.sum(avalue.labeltally_without) != num_neighbors and t2 is not None)  # type: ignore
                        or (np.sum(avalue.labeltally_without) >= num_neighbors and t2 is None)  # type: ignore
                    ):
                        continue
                    label_with = np.argmax(avalue.labeltally_with)  # type: ignore
                    label_without = np.argmax(avalue.labeltally_without)  # type: ignore
                    utility_diff = (
                        label_utilities[label_with, j] - label_utilities[label_without, j]
                        if t2 is not None
                        else label_utilities[label_with, j] - null_scores[j]
                    )
                    all_importances[i] += (1.0 / comb(n_units - 1, avalue.tupletally)) * rvalue * utility_diff
    all_importances /= n_units * n_test
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
        self.X: Optional[NDArray] = None
        self.y: Optional[NDArray] = None
        self.provenance: Optional[Provenance] = None
        self.randomstate = np.random.RandomState(seed)
        self.label_encoder = LabelEncoder()
        self.logger = logger if logger is not None else getLogger(__name__)

    def _fit(self, X: NDArray, y: NDArray, provenance: Provenance) -> "ShapleyImportance":
        self.X = X
        self.y = self.label_encoder.fit_transform(y)
        self.provenance = provenance
        return self

    def _score(
        self,
        X: NDArray,
        y: Optional[NDArray] = None,
        units: Union[Sequence[Hashable], NDArray[np.int32], None] = None,
        world: Union[Sequence[Hashable], NDArray[np.int32], None] = None,
        **kwargs
    ) -> Iterable[float]:
        if self.X is None or self.y is None or self.provenance is None:
            raise ValueError("The fit function was not called first.")
        if y is None:
            raise ValueError("The 'y' argument cannot be None.")
        else:
            y = self.label_encoder.transform(y)
            assert y is not None

        if units is None:
            units = np.arange(self.provenance.num_units)
        elif not isinstance(units, np.ndarray):
            units = np.array([self.provenance.units_index[x] for x in units], dtype=int)
        if world is None:
            world = np.ones_like(units, dtype=int)
        elif not isinstance(world, np.ndarray):
            world = np.array([self.provenance.candidates_index[x] for x in world], dtype=int)
        return self._shapley(self.X, self.y, X, y, self.provenance, units, world)

    def _shapley(
        self,
        X: NDArray,
        y: NDArray,
        X_test: NDArray,
        y_test: NDArray,
        provenance: Provenance,
        units: NDArray[np.int32],
        world: NDArray[np.int32],
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
        X: NDArray,
        y: NDArray,
        X_test: NDArray,
        y_test: NDArray,
        provenance: Provenance,
        units: NDArray[np.int32],
        world: NDArray[np.int32],
    ) -> Iterable[float]:
        # if checknan(provenance):  # TODO: Remove this hack.
        #     units = np.arange(X.shape[0])
        #     world = np.zeros_like(units, dtype=int)
        #     provenance = np.arange(X.shape[0])
        #     provenance = reshape(provenance)

        # # Convert provenance and units to bit-arrays.
        # provenance = binarize(provenance)

        # Apply the feature extraction pipeline if it was provided.
        if self.pipeline is not None:
            self.pipeline.fit(X, y=y)
            X_test = self.pipeline.transform(X_test)

        # Compute null score.
        null_score = self.utility.null_score(X, y, X_test, y_test)

        # Iterate over all subsets of units.
        # n_units_total = provenance.shape[2]
        # n_candidates_total = provenance.shape[3]
        n_units = len(units)
        importance = np.zeros(n_units)
        for iteration in product(*[[0, world[i]] for i in range(n_units)]):
            iter = np.array(iteration, dtype=int)
            s_iter = np.sum(iter)

            # Get indices of data points selected based on the iteration query.
            # query = np.zeros((n_units_total, n_candidates_total))
            # query[units, world] = iter
            indices = provenance.query(iter)

            # Train the model and score it. If we fail at any step we get zero score.
            score = null_score
            with warnings.catch_warnings():
                warnings.simplefilter("error", category=RuntimeWarning)
                warnings.simplefilter("error", category=UserWarning)
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
        X: NDArray,
        y: NDArray,
        X_test: NDArray,
        y_test: NDArray,
        provenance: Provenance,
        units: NDArray[np.int32],
        world: NDArray[np.int32],
        iterations: int = DEFAULT_MC_ITERATIONS,
        timeout: int = DEFAULT_MC_TIMEOUT,
        tolerance: float = DEFAULT_MC_TOLERANCE,
        truncation_steps: int = DEFAULT_MC_TRUNCATION_STEPS,
    ) -> Iterable[float]:
        # if checknan(provenance):  # TODO: Remove this hack.
        #     units = np.arange(X.shape[0])
        #     world = np.zeros_like(units, dtype=int)
        #     provenance = np.arange(X.shape[0])
        #     provenance = reshape(provenance)

        # # Convert provenance and units to bit-arrays.
        # provenance = binarize(provenance)

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
        n_units_total = provenance.num_units
        # n_candidates_total = provenance.shape[3]
        n_units = len(units)
        # simple_provenance = self._simple_provenance or bool(
        #     provenance.shape[1] == 1
        #     and provenance.shape[3] == 1
        #     and provenance.shape[0] == provenance.shape[2]
        #     and np.all(np.equal(provenance[:, 0, :, 0], np.eye(n_units)))
        # )
        all_importances = np.zeros((n_units, iterations))
        all_truncations = np.ones(iterations, dtype=int) * n_units
        start_time = time.time()
        for i in range(iterations):
            idxs = self.randomstate.permutation(n_units)
            importance = np.zeros(n_units)
            query = np.zeros((n_units_total,), dtype=int)

            new_score = null_score
            truncation_counter = 0

            for j, idx in enumerate(idxs):
                old_score = new_score

                # Get indices of data points selected based on the iteration query.
                query[units[idx]] = world[idx]
                indices = provenance.query(query)
                # indices = get_indices(provenance, query, simple_provenance=simple_provenance)

                # Train the model and score it. If we fail at any step we get zero score.
                new_score = null_score
                with warnings.catch_warnings():
                    warnings.simplefilter("error", category=RuntimeWarning)
                    warnings.simplefilter("error", category=UserWarning)
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
        X: NDArray,
        y: NDArray,
        X_test: NDArray,
        y_test: NDArray,
        provenance: Provenance,
        units: NDArray[np.int32],
        world: NDArray[np.int32],
        k: int,
        distance: DistanceCallable,
    ) -> Iterable[float]:
        if provenance.max_disjunctions > 1:
            raise ValueError(
                "The neighbor method cannot be applied to data with provenance polynomials containing disjunctions."
            )

        # Apply the feature extraction pipeline if it was provided.
        if self.pipeline is not None:
            X = self.pipeline.fit_transform(X, y=y)
            X_test = self.pipeline.transform(X_test)

        # Ensure X and X_test are not sparse.
        if issparse(X):
            assert isinstance(X, spmatrix)
            X = X.todense()
        if issparse(X_test):
            assert isinstance(X_test, spmatrix)
            X_test = X_test.todense()

        # Convert matrices to instances of ndarray.
        if isinstance(X, np.matrix):
            X = np.asarray(X)
        if isinstance(X_test, np.matrix):
            X_test = np.asarray(X_test)

        # Compute the size of the test batch to split the X_test matrix if if X and X_test are too large.
        n_train, n_test, n_units = X.shape[0], X_test.shape[0], units.shape[0]
        batch_size = get_test_batch_size(n_train, n_test)
        all_importances = np.zeros((n_units), dtype=float)

        # We iterate over all batches of test data and compute importances.
        for start in range(0, n_test, batch_size):
            X_test_batch = X_test[start : start + batch_size]  # noqa: E203
            n_test_batch = X_test_batch.shape[0]

            # Compute the distances between training and text data examples.
            distances = distance(X, X_test_batch)

            # Compute the utilitiy values between training and test labels.
            utilities = self.utility.elementwise_score(X_train=X, y_train=y, X_test=X_test_batch, y_test=y_test)

            # Compute null scores.
            null_scores = self.utility.elementwise_null_score(X, y, X_test_batch, y_test)

            cur_importances: NDArray
            if k == 1 and provenance.max_conjunctions == 1:
                cur_importances = compute_shapley_1nn_mapfork(
                    y,
                    distances,
                    utilities,
                    provenance,
                    units,
                    world,
                    null_scores=null_scores,
                )
            else:
                cur_importances = compute_shapley_add(
                    y,
                    distances,
                    utilities,
                    provenance,
                    units,
                    world,
                    num_neighbors=k,
                    num_classes=len(self.label_encoder.classes_),
                    null_scores=null_scores,
                )

            all_importances += cur_importances * (float(n_test_batch) / float(n_test))

        return all_importances
