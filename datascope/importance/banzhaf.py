import numpy as np
import time
import warnings
from itertools import product

# from numba import prange, jit
from numpy import ndarray
from sklearn.pipeline import Pipeline

from typing_extensions import Literal
from typing import Optional, Iterable

from .common import DEFAULT_SEED, Utility, binarize, get_indices, reshape
from .importance import Importance
from .shapley import ImportanceMethod, DEFAULT_MC_ITERATIONS, DEFAULT_MC_TIMEOUT, DEFAULT_MC_TOLERANCE, DEFAULT_MC_TRUNCATION_STEPS, DEFAULT_SEED, checknan
from loguru import logger

class BanzhafImportance(Importance):
    def __init__(
        self,
        method: Literal[ImportanceMethod.BRUTEFORCE, ImportanceMethod.MONTECARLO],
        utility: Utility,
        pipeline: Optional[Pipeline] = None,
        mc_iterations: int = DEFAULT_MC_ITERATIONS,
        mc_timeout: int = DEFAULT_MC_TIMEOUT,
        mc_tolerance: float = DEFAULT_MC_TOLERANCE,
        mc_truncation_steps: int = DEFAULT_MC_TRUNCATION_STEPS,
        mc_preextract: bool = False,
        seed: int = DEFAULT_SEED,
    ):
        self.method = ImportanceMethod(method)
        self.utility = utility
        self.pipeline = pipeline
        self.mc_iterations = mc_iterations
        self.mc_timeout = mc_timeout
        self.mc_tolerance = mc_tolerance
        self.mc_truncation_steps = mc_truncation_steps
        self.mc_preextract = mc_preextract
        self.X: Optional[ndarray] = None
        self.y: Optional[ndarray] = None
        self.provenance: Optional[ndarray] = None
        self._simple_provenance = False
        self.randomstate = np.random.RandomState(seed)
    
    def _fit(self, X: ndarray, y: ndarray, provenance: Optional[ndarray] = None) -> "BanzhafImportance":
        self.X = X
        self.y = y

        if provenance is None:
            provenance = np.arange(X.shape[0])
            self._simple_provenance = True
        if checknan(provenance):
            self._simple_provenance = True
        if not checknan(provenance):
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
        return self._banzhaf(self.X, self.y, X, y, self.provenance, units, world)

    def _banzhaf(
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
            return self._banzhaf_bruteforce(X, y, X_test, y_test, provenance, units, world)
        
        elif self.method == ImportanceMethod.MONTECARLO:
            return self._banzhaf_montecarlo(
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
        
        else:
            raise NotImplementedError("Only Monte Carlo is implemented")
    
    def _banzhaf_bruteforce(
        self,
        X: ndarray,
        y: ndarray,
        X_test: ndarray,
        y_test: ndarray,
        provenance: ndarray,
        units: ndarray,
        world: ndarray,
    ) -> Iterable[float]:

        if checknan(provenance):
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
                    #print(X_train)
                    #print(y_train)
                    if self.pipeline is not None:
                        X_train = self.pipeline.fit_transform(X_train, y=y)
                    score = self.utility(X_train, y_train, X_test, y_test, null_score)
                    
                except (ValueError, RuntimeWarning, UserWarning):
                    pass

            # Compute the factor and update the Shapley values of respective units.
            # score is the difference between the null score and the score of the current iteration.
            importance[np.where(iter==1)] += (1/ (2**len(units))) * score
            importance[np.where(iter==0)] -= (1/ (2**len(units))) * score
        
        return importance

    def _banzhaf_montecarlo(
        self,
        X: ndarray,
        y: ndarray,
        X_test: ndarray,
        y_test: ndarray,
        provenance: ndarray,
        units: ndarray,
        world: ndarray,
        iterations: int = 10 * DEFAULT_MC_ITERATIONS,
        timeout: int = DEFAULT_MC_TIMEOUT,
        tolerance: float = DEFAULT_MC_TOLERANCE,
        truncation_steps: int = DEFAULT_MC_TRUNCATION_STEPS,
    ) -> Iterable[float]:
        if checknan(provenance):
            units = np.arange(X.shape[0])
            world = np.zeros_like(units, dtype=int)
            provenance = np.arange(X.shape[0])
            provenance = reshape(provenance)
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
            X = X_tr

        # Run a given number of iterations.
        n_units_total = provenance.shape[2]
        n_candidates_total = provenance.shape[3]
        n_units = len(units)
        simple_provenance = self._simple_provenance or bool(
            provenance.shape[1] == 1 and provenance.shape[3] == 1 and np.all(np.equal(provenance[:, 0, :, 0],np.eye(n_units)))
        )
        all_importances = np.zeros((n_units, iterations))
        start_time = time.time()
        list_of_queries = []
        for i in range(iterations):
            #sample_size = self.randomstate.randint(low=0, high=n_units+1)
            #idxs = self.randomstate.choice(n_units, size=sample_size, replace=False)
            importance = np.zeros(n_units)
            query = np.zeros((n_units_total, n_candidates_total))
            new_score = null_score
            truncation_counter = 0
            for idx in range(n_units):
                old_score = new_score
                # Get indices of data points selected based on the iteration query.
                query[units[idx], world[idx]] = self.randomstate.randint(low=2)
            indices = get_indices(provenance, query, simple_provenance=simple_provenance)
            list_of_queries.append(query[:, 0])
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
                # print(query[:, 0])
            importance[np.where(query[:, 0]==1)] += new_score
            importance[np.where(query[:, 0]==0)] -= new_score

            if np.abs(new_score - mean_score) <= np.abs(tolerance * mean_score):
                truncation_counter += 1
                if truncation_steps > 0 and truncation_counter > truncation_steps:
                    logger.warning("Truncation reached")
                    break
            else:
                truncation_counter = 0

            all_importances[:, i] = importance

            # Check if we have timed out.
            elapsed_time = time.time() - start_time
            if timeout > 0 and elapsed_time > timeout:
                logger.warning("Timeout reached. Stopping.")
                all_importances = all_importances[:, :i]
                break
        # count how many true in first column
        list_of_queries = np.array(list_of_queries)
        scores = np.average(all_importances, axis=1)
        return scores