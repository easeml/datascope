from abc import abstractmethod
import numpy as np

from abc import ABC
from collections import defaultdict
from numpy import ndarray
from typing import Optional, Protocol, Callable, Union
from sklearn.base import clone
from sklearn.metrics import accuracy_score


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

    @abstractmethod
    def elementwise_score(
        self,
        X_train: ndarray,
        y_train: ndarray,
        X_test: ndarray,
        y_test: ndarray,
    ) -> ndarray:
        raise NotImplementedError("This utility does not implement elementwise scoring.")


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
        score = 0.0
        try:
            # TODO: Ensure fit clears the model.
            np.random.seed(seed)
            model = self._model_fit(self.model, X_train, y_train)
            y_pred = self._model_predict(model, X_test)
            score = self._metric_score(self.metric, y_test, y_pred)
        except ValueError:
            return null_score if null_score is not None else self.null_score(X_train, y_train, X_test, y_test)
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


class SklearnModelAccuracy(SklearnModelUtility):
    def __init__(self, model: SklearnModel) -> None:
        super().__init__(model, accuracy_score)

    def elementwise_score(
        self,
        X_train: ndarray,
        y_train: ndarray,
        X_test: ndarray,
        y_test: ndarray,
    ) -> ndarray:
        return np.equal.outer(y_train, y_test).astype(float)


class SklearnModelEqualizedOddsDifference(SklearnModelUtility):
    def __init__(self, model: SklearnModel, sensitive_feature: int) -> None:
        super().__init__(model, metric)


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


def reshape(provenance: ndarray) -> ndarray:
    provenance = pad_jagged_array(provenance, fill_value=-1)
    result = provenance
    if result.ndim == 1:
        result = result.reshape((-1, 1, 1, 1))
    elif result.ndim == 2:
        result = result.reshape((result.shape[0], 1, result.shape[1], 1))
    elif result.ndim == 3:
        result = result.reshape(result.shape + (1,))
    if result.ndim != 4:
        raise ValueError("Cannot reshape the provenance array with shape %s." % str(provenance.shape))
    if result.shape[3] == 1:
        result = np.concatenate([result, np.zeros_like(result)], axis=3)
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


def binarize(provenance: ndarray) -> ndarray:
    assert provenance.ndim == 4
    assert provenance.shape[-1] == 2
    umax = provenance[..., 0].max()
    cmax = provenance[..., 1].max()
    resultshape = provenance.shape[:2] + (umax + 1, cmax + 1)
    result = np.zeros(resultshape, dtype=int)
    for idx, unit in np.ndenumerate(provenance[..., 0]):
        candidate = provenance[idx[:3] + (1,)]
        if unit != -1 and candidate != -1:
            ridx = idx[:2] + (unit, candidate)
            result[ridx] = 1
    return result


def get_indices(provenance: ndarray, query: ndarray, simple_provenance: bool = False) -> ndarray:
    assert provenance.ndim == 4
    assert query.ndim in [1, 2]
    if query.ndim == 1:
        query = np.broadcast_to(query[:, np.newaxis], query.shape + (provenance.shape[-1],))

    n_units = query.shape[0]
    if simple_provenance or (
        provenance.shape[1] == 1 and provenance.shape[3] == 1 and np.all(provenance[:, 0, :, 0] == np.eye(n_units))
    ):
        return query[:, 0].astype(bool)
    # newshape = provenance.shape[:1] + tuple(1 for _ in range(3 - provenance.ndim)) + provenance.shape[1:]
    # provenance = provenance.reshape(newshape)
    query = np.broadcast_to(query, provenance.shape)
    land = np.logical_and(provenance, query)
    eq = np.equal(land, provenance)
    a1 = np.all(eq, axis=(2, 3))
    a2 = np.any(a1, axis=1)
    return a2


class Provenance(ndarray):
    def __new__(cls, a):
        obj = np.asarray(a).view(cls)
        cls._validate(obj)
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        Provenance._validate(obj)

    @classmethod
    def _validate(cls, obj: "Provenance") -> None:
        if obj.ndim != 4:
            raise ValueError("The provided provenance array must have 4 dimensions.")

    def get_indices(self, query: ndarray) -> ndarray:
        assert query.ndim in [1, 2]
        if query.ndim == 1:
            query = np.broadcast_to(query[:, np.newaxis], query.shape + (self.shape[-1],))
        query = np.broadcast_to(query, self.shape)
        land = np.logical_and(self, query)
        eq = np.equal(land, self)
        a1 = np.all(eq, axis=(2, 3))
        a2 = np.any(a1, axis=1)
        return a2

    def indices(self, units: Optional[ndarray] = None, world: Optional[Union[ndarray, int]] = None) -> ndarray:
        if units is None:
            if world is None:
                return np.ones(self.shape[0], dtype=int)
            else:
                if isinstance(world, int):
                    world = np.repeat(world, self.shape[0])
                worldslice = self[np.arange(self.shape[0]), :, :, world]
                return np.any(worldslice, axis=(1, 2))
        else:
            if world is None:
                return self.get_indices(units)
            else:
                query = np.zeros(self.shape[2:], dtype=int)
                query[units.astype(bool), world] = np.ones(self.shape)
                return self.get_indices(query)

    def expand(
        self,
        tuples: Optional[int] = None,
        disjunctions: Optional[int] = None,
        units: Optional[int] = None,
        candidates: Optional[int] = None,
    ) -> "Provenance":
        if tuples is None and disjunctions is None and units is None and candidates is None:
            return self

        pad_width = [[0, 0] for _ in range(4)]
        if tuples is not None:
            pad_width[0] = [0, tuples]
        if disjunctions is not None:
            pad_width[1] = [0, disjunctions]
        if units is not None:
            pad_width[2] = [0, units]
        if candidates is not None:
            pad_width[3] = [0, candidates]
        result: ndarray = np.pad(self, pad_width=pad_width)
        return result.view(Provenance)
