from abc import abstractmethod
import numpy as np

from abc import ABC
from collections import defaultdict
from numpy import ndarray
from typing import Optional, Protocol, Callable
from sklearn.base import clone


class SklearnModel(Protocol):
    def fit(self, X: ndarray, y: ndarray, sample_weight: Optional[ndarray] = None) -> None:
        pass

    def predict(self, X: ndarray) -> ndarray:
        pass


MetricCallable = Callable[[ndarray, ndarray], float]
DistanceCallable = Callable[[ndarray, ndarray], ndarray]


DEFAULT_SCORING_MAXITER = 100
DEFAULT_SEED = 7


class Utility(ABC):
    @abstractmethod
    def __call__(
        self,
        X_train: ndarray,
        y_train: ndarray,
        X_test: ndarray,
        y_test: ndarray,
        null_score: Optional[float] = None,
        seed: int = DEFAULT_SEED,
    ) -> float:
        pass

    @abstractmethod
    def null_score(
        self,
        X_train: ndarray,
        y_train: ndarray,
        X_test: ndarray,
        y_test: ndarray,
        maxiter: int = DEFAULT_SCORING_MAXITER,
        seed: int = DEFAULT_SEED,
    ) -> float:
        pass

    @abstractmethod
    def mean_score(
        self,
        X_train: ndarray,
        y_train: ndarray,
        X_test: ndarray,
        y_test: ndarray,
        maxiter: int = DEFAULT_SCORING_MAXITER,
        seed: int = DEFAULT_SEED,
    ) -> float:
        pass


class SklearnModelUtility(Utility):
    def __init__(self, model: SklearnModel, metric: MetricCallable) -> None:
        self.model = model
        self.metric = metric

    def __call__(
        self,
        X_train: ndarray,
        y_train: ndarray,
        X_test: ndarray,
        y_test: ndarray,
        null_score: Optional[float] = None,
        seed: int = DEFAULT_SEED,
    ) -> float:
        score = null_score if null_score is not None else self.null_score(X_train, y_train, X_test, y_test)
        try:
            # TODO: Ensure fit clears the model.
            np.random.seed(seed)
            model = self._model_fit(self.model, X_train, y_train)
            y_pred = self._model_predict(model, X_test)
            score = self._metric_score(self.metric, y_test, y_pred)
        except ValueError:
            pass
        return score

    def _model_fit(self, model: SklearnModel, X_train: ndarray, y_train: ndarray) -> SklearnModel:
        model = clone(model)
        model.fit(X_train, y_train)
        return model

    def _model_predict(self, model: SklearnModel, X_test: ndarray) -> ndarray:
        return model.predict(X_test)

    def _metric_score(self, metric: MetricCallable, y_test: ndarray, y_pred: ndarray) -> float:
        return metric(y_test, y_pred)

    def null_score(
        self,
        X_train: ndarray,
        y_train: ndarray,
        X_test: ndarray,
        y_test: ndarray,
        maxiter: int = DEFAULT_SCORING_MAXITER,
        seed: int = DEFAULT_SEED,
    ) -> float:
        np.random.seed(seed)
        model = self._model_fit(self.model, X_train, y_train)
        y_pred = self._model_predict(model, X_test)
        scores = []
        np.random.seed(seed)
        for _ in range(maxiter):
            idx = np.random.randint(len(y_test))
            y_spoof = np.repeat(y_pred[[idx]], len(y_test), axis=0)
            scores.append(self._metric_score(self.metric, y_test, y_spoof))
        return min(scores)

    def mean_score(
        self,
        X_train: ndarray,
        y_train: ndarray,
        X_test: ndarray,
        y_test: ndarray,
        maxiter: int = DEFAULT_SCORING_MAXITER,
        seed: int = DEFAULT_SEED,
    ) -> float:
        np.random.seed(seed)
        model = self._model_fit(self.model, X_train, y_train)
        y_pred = self._model_predict(model, X_test)
        scores = []
        np.random.seed(seed)
        for _ in range(maxiter):
            idx = np.random.choice(len(y_test), int(len(y_test) * 0.5))
            scores.append(self._metric_score(self.metric, y_test[idx], y_pred[idx]))
        return np.mean(scores)


def get_dimensions(array, level=0):
    yield level, len(array)
    try:
        for row in array:
            yield from get_dimensions(row, level + 1)
    except TypeError:  # not an iterable
        pass


def get_max_shape(array):
    dimensions = defaultdict(int)
    for level, length in get_dimensions(array):
        dimensions[level] = max(dimensions[level], length)
    return [value for _, value in sorted(dimensions.items())]


def iterate_nested_array(array, index=()):
    try:
        for idx, row in enumerate(array):
            yield from iterate_nested_array(row, (*index, idx))
    except TypeError:  # final level
        yield (*index, slice(len(array))), array


def pad_jagged_array(array, fill_value):
    dimensions = get_max_shape(array)
    result = np.full(dimensions, fill_value)
    for index, value in iterate_nested_array(array):
        result[index] = value
    return result


def one_hot_encode(array: ndarray, mergelast: bool = False) -> ndarray:
    vmax = array.max()
    resultshape = (array.shape[:-1] if mergelast else array.shape) + (vmax + 1,)
    result = np.zeros(resultshape, dtype=int)
    indicator = np.eye(vmax + 1, dtype=int)
    if mergelast:
        for idx, row in np.ndenumerate(array):
            result[idx[:-1]] |= indicator[row]
    else:
        for idx, row in np.ndenumerate(array):
            result[idx] = indicator[row]
    return result


def get_indices(provenance: ndarray, query: ndarray) -> ndarray:
    assert provenance.ndim >= 2
    newshape = provenance.shape[:1] + tuple(1 for _ in range(3 - provenance.ndim)) + provenance.shape[1:]
    provenance = provenance.reshape(newshape)
    query = np.broadcast_to(query, newshape)
    land = np.logical_and(provenance, query)
    eq = np.equal(land, provenance)
    a1 = np.all(eq, axis=2)
    a2 = np.any(a1, axis=1)
    return a2
