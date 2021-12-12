import math
import numpy as np

from enum import Enum
from itertools import product
from math import comb
from numpy import ndarray
from sklearn.neighbors import DistanceMetric
from typing import Literal, Optional, Iterable, Tuple, Dict

from .common import DEFAULT_SEED, DistanceCallable, Utility, get_indices, one_hot_encode, pad_jagged_array
from .importance import Importance


class ImportanceMethod(Enum):
    BRUTEFORCE = "bruteforce"
    MONTECARLO = "montecarlo"
    NEIGHBOR = "neighbor"


DEFAULT_MC_ITERATIONS = 500
DEFAULT_MC_TOLERANCE = 0.1
DEFAULT_MC_TRUNCATION_STEPS = 5
DEFAULT_NN_K = 1
DEFAULT_NN_DISTANCE = DistanceMetric.get_metric("minkowski").pairwise


class ADD:
    def __init__(self, size: int, ashape: Tuple[int]) -> None:
        self.size = size
        self.ashape = ashape
        self.nodes = np.zeros(size)
        self.cleft = np.zeros(size)
        self.cright = np.zeros(size)
        self.aleft = np.zeros(ashape)
        self.aright = np.zeros(ashape)

    def restrict(self, variable: int, value: bool) -> "ADD":
        pass

    def sum(self, other: "ADD") -> "ADD":
        pass

    def append(self, other: "ADD", path: Optional[Dict[int, bool]]) -> "ADD":
        pass

    def modelcount(self) -> ndarray:
        pass

    @classmethod
    def construct_tree(cls, variables: Iterable[int]) -> "ADD":
        pass

    @classmethod
    def construct_chain(cls, variables: Iterable[int]) -> "ADD":
        pass


def compile_add(sigma: ndarray, provenance: ndarray, units: ndarray, btuple: int) -> ADD:
    pass


def compute_shapley_1nn(
    distances: ndarray,
    utilities: ndarray,
    provenance: ndarray,
    units: ndarray,
) -> Iterable[float]:

    if not np.all(provenance.sum(axis=1) == 1):
        raise ValueError("The provenance of all data examples must reference at most one unit.")

    n_test = distances.shape[1]
    n_units = len(units)

    # Compute the minimal distance for each unit and each test example.
    unit_provenance = provenance[:, units].astype(bool)
    unit_distances = np.zeros((n_units, n_test))  # TODO: Make this faster.
    unit_utilities = np.zeros((n_units, n_test))  # TODO: Make this faster.
    for i in range(n_units):
        for j in range(n_test):
            idx = np.argmin(distances[unit_provenance[:, i], j])
            unit_distances[i, j] = distances[unit_provenance[:, i]][idx, j]
            unit_utilities[i, j] = utilities[unit_provenance[:, i]][idx, j]

    # Compute unit importances.
    all_importances = np.zeros((n_units + 1, n_test))
    unit_utilities = np.vstack([unit_utilities, np.zeros((1, n_test))])
    for j in range(n_test):
        idxs = np.append(np.argsort(unit_distances[:, j]), [n_units])
        for i in reversed(range(n_units)):
            all_importances[idxs[i], j] = all_importances[idxs[i + 1], j] + (
                unit_utilities[idxs[i], j] - unit_utilities[idxs[i + 1], j]
            ) * math.comb(n_units - i, i + 1)

    # Aggregate results.
    importances = np.mean(all_importances[:-1], axis=1)
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

    def _fit(self, X: ndarray, y: ndarray, provenance: Optional[ndarray] = None) -> None:
        self.X = X
        self.y = y
        if provenance is None:
            provenance = np.arange(len(X))
        if provenance.ndim == 1:
            provenance = provenance.reshape((-1, 1))
        self.provenance = pad_jagged_array(provenance, fill_value=-1)

    def _score(self, X: ndarray, y: Optional[ndarray] = None, **kwargs) -> Iterable[float]:
        if self.X is None or self.y is None or self.provenance is None:
            raise ValueError("The fit function was not called first.")
        if y is None:
            raise ValueError("The 'y' argument cannot be None.")

        units = kwargs.get("units", np.unique(self.provenance))
        units = np.delete(units, np.where(units == -1))
        return self._shapley(self.X, self.y, X, y, self.provenance, units)

    def _shapley(
        self, X: ndarray, y: ndarray, X_test: ndarray, y_test: ndarray, provenance: ndarray, units: ndarray
    ) -> Iterable[float]:
        if self.method == ImportanceMethod.BRUTEFORCE:
            return self._shapley_bruteforce(X, y, X_test, y_test, provenance, units)
        elif self.method == ImportanceMethod.MONTECARLO:
            return self._shapley_montecarlo(
                X, y, X_test, y_test, provenance, units, self.mc_iterations, self.mc_tolerance, self.mc_truncation_steps
            )
        elif self.method == ImportanceMethod.NEIGHBOR:
            return self._shapley_neighbor(X, y, X_test, y_test, provenance, units, self.nn_k, self.nn_distance)
        else:
            raise ValueError("Unknown method '%s'." % self.method)

    def _shapley_bruteforce(
        self, X: ndarray, y: ndarray, X_test: ndarray, y_test: ndarray, provenance: ndarray, units: ndarray
    ) -> Iterable[float]:

        # Convert provenance and units to bit-arrays.
        provenance = one_hot_encode(provenance, mergelast=True)

        # Compute null score.
        null_score = self.utility.null_score(X, y, X_test, y_test)

        # Iterate over all subsets of units.
        n_units_total = provenance.shape[-1]
        n_units = len(units)
        importance = np.zeros(n_units)
        for iteration in product(*[[0, 1] for _ in range(n_units)]):
            iter = np.array(iteration)
            s_iter = np.sum(iter)

            # Get indices of data points selected based on the iteration query.
            query = np.zeros(n_units_total)
            query[units] = iter
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
        iterations: int = DEFAULT_MC_ITERATIONS,
        tolerance: float = DEFAULT_MC_TOLERANCE,
        truncation_steps: int = DEFAULT_MC_TRUNCATION_STEPS,
    ) -> Iterable[float]:

        # Convert provenance and units to bit-arrays.
        provenance = one_hot_encode(provenance, mergelast=True)

        # Compute mean score.
        null_score = self.utility.null_score(X, y, X_test, y_test)
        mean_score = self.utility.mean_score(X, y, X_test, y_test)

        # Run a given number of iterations.
        n_units_total = provenance.shape[-1]
        n_units = len(units)
        truncation_counter = 0
        all_importances = np.zeros((n_units, iterations))
        for i in range(iterations):
            idxs = self.randomstate.permutation(n_units)
            importance = np.zeros(n_units)
            query = np.zeros(n_units_total)

            new_score = null_score

            for idx in idxs:
                old_score = new_score

                # Get indices of data points selected based on the iteration query.
                query[units[idx]] = 1
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
        k: int,
        distance: DistanceCallable,
    ) -> Iterable[float]:

        # Convert provenance and units to bit-arrays.
        provenance = one_hot_encode(provenance, mergelast=True)

        # Compute the distances between training and text data examples.
        distances = distance(X, X_test)

        # Compute the utilitiy values between training and test labels.
        # TODO: Make this faster.
        # TODO: Enable different element-wise utilities.
        utilities = np.zeros((len(y), len(y_test)))
        for i in range(len(y)):
            for j in range(len(y_test)):
                utilities[i, j] = float(y[i] == y_test[j])

        if k == 1:
            return compute_shapley_1nn(distances, utilities, provenance, units)
        else:
            raise ValueError("The value '%d' for the k-parameter is not possible.")
