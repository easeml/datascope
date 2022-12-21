import collections.abc
import numpy as np
import warnings

from abc import ABC, abstractmethod
from collections import defaultdict
from numpy import ndarray
from typing import Iterable, Optional, Callable, Sequence, Tuple, Union, List
from typing_extensions import Protocol
from sklearn.base import clone
from sklearn.metrics import accuracy_score


class SklearnModel(Protocol):
    def fit(self, X: ndarray, y: ndarray, sample_weight: Optional[ndarray] = None) -> None:
        pass

    def predict(self, X: ndarray) -> ndarray:
        pass


class SklearnTransformer(Protocol):
    def fit(self, X: ndarray, y: ndarray, sample_weight: Optional[ndarray] = None) -> None:
        pass

    def transform(self, X: ndarray) -> ndarray:
        pass


class SklearnPipeline(SklearnModel, SklearnTransformer):
    steps: List[Tuple[str, Union[SklearnModel, SklearnTransformer]]]


SklearnModelOrPipeline = Union[SklearnModel, SklearnPipeline]


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

    def elementwise_score(
        self,
        X_train: ndarray,
        y_train: ndarray,
        X_test: ndarray,
        y_test: ndarray,
    ) -> ndarray:
        raise NotImplementedError("This utility does not implement elementwise scoring.")

    def elementwise_null_score(
        self,
        X_train: ndarray,
        y_train: ndarray,
        X_test: ndarray,
        y_test: ndarray,
    ) -> ndarray:
        raise NotImplementedError("This utility does not implement elementwise scoring.")


class JointUtility(Utility):
    def __init__(self, *utilities: Utility, weights: Optional[Iterable[float]] = None) -> None:
        self._utilities = list(utilities)
        if weights is None:
            self._weights = [1.0 / len(self._utilities) for _ in range(len(self._utilities))]
        else:
            self._weights = list(weights)
            if len(self._weights) != len(self._utilities):
                raise ValueError("The list of weights must be the same length as the list of utilities.")
            # s = sum(self._weights)
            # self._weights = [w / s for w in self._weights]
        super().__init__()

    def __call__(
        self,
        X_train: ndarray,
        y_train: ndarray,
        X_test: ndarray,
        y_test: ndarray,
        null_score: Optional[float] = None,
        seed: int = DEFAULT_SEED,
    ) -> float:
        scores = [u(X_train, y_train, X_test, y_test, null_score=np.nan, seed=seed) for u in self._utilities]
        if any(np.isnan(x) for x in scores):
            return null_score if null_score is not None else self.null_score(X_train, y_train, X_test, y_test)
        return sum(w * s for w, s in zip(self._weights, scores))

    def null_score(
        self,
        X_train: ndarray,
        y_train: ndarray,
        X_test: ndarray,
        y_test: ndarray,
    ) -> float:
        return sum(w * u.null_score(X_train, y_train, X_test, y_test) for w, u in zip(self._weights, self._utilities))

    def mean_score(
        self,
        X_train: ndarray,
        y_train: ndarray,
        X_test: ndarray,
        y_test: ndarray,
        maxiter: int = DEFAULT_SCORING_MAXITER,
        seed: int = DEFAULT_SEED,
    ) -> float:
        return sum(
            w * u.mean_score(X_train, y_train, X_test, y_test, maxiter=maxiter, seed=seed)
            for w, u in zip(self._weights, self._utilities)
        )

    def elementwise_score(
        self,
        X_train: ndarray,
        y_train: ndarray,
        X_test: ndarray,
        y_test: ndarray,
    ) -> ndarray:
        scores = np.stack(
            [w * u.elementwise_score(X_train, y_train, X_test, y_test) for w, u in zip(self._weights, self._utilities)]
        )
        return np.sum(scores, axis=0)

    def elementwise_null_score(
        self,
        X_train: ndarray,
        y_train: ndarray,
        X_test: ndarray,
        y_test: ndarray,
    ) -> ndarray:
        scores = np.stack(
            [
                w * u.elementwise_null_score(X_train, y_train, X_test, y_test)
                for w, u in zip(self._weights, self._utilities)
            ]
        )
        return np.sum(scores, axis=0)


class SklearnModelUtility(Utility):
    def __init__(self, model: SklearnModelOrPipeline, metric: Optional[MetricCallable]) -> None:
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
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            try:
                # TODO: Ensure fit clears the model.
                np.random.seed(seed)
                model = self._model_fit(self.model, X_train, y_train)
                y_pred = self._model_predict(model, X_test)
                score = self._metric_score(self.metric, y_test, y_pred, X_test=X_test)
            except (ValueError, RuntimeWarning):
                try:
                    return null_score if null_score is not None else self.null_score(X_train, y_train, X_test, y_test)
                except (ValueError, RuntimeWarning):
                    return score
        return score

    def _model_fit(self, model: SklearnModelOrPipeline, X_train: ndarray, y_train: ndarray) -> SklearnModelOrPipeline:
        model = clone(model)
        model.fit(X_train, y_train)
        return model

    def _model_predict(self, model: SklearnModelOrPipeline, X_test: ndarray) -> ndarray:
        return model.predict(X_test)

    def _metric_score(
        self,
        metric: Optional[MetricCallable],
        y_test: ndarray,
        y_pred: ndarray,
        *,
        X_test: Optional[ndarray] = None,
        indices: Optional[ndarray] = None,
    ) -> float:
        if metric is None:
            raise ValueError("The metric was not provided.")
        return metric(y_test, y_pred)

    def null_score(
        self,
        X_train: ndarray,
        y_train: ndarray,
        X_test: ndarray,
        y_test: ndarray,
    ) -> float:
        scores = []
        for x in np.unique(y_train):
            y_spoof = np.full_like(y_test, x, dtype=y_test.dtype)
            scores.append(self._metric_score(self.metric, y_test, y_spoof, X_test=X_test))

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
            scores.append(self._metric_score(self.metric, y_test[idx], y_pred[idx], X_test=X_test[idx, :], indices=idx))
        return np.mean(scores).item()


class SklearnModelAccuracy(SklearnModelUtility):
    def __init__(self, model: SklearnModelOrPipeline) -> None:
        super().__init__(model, accuracy_score)

    def elementwise_score(
        self,
        X_train: ndarray,
        y_train: ndarray,
        X_test: ndarray,
        y_test: ndarray,
    ) -> ndarray:
        return np.equal.outer(y_train, y_test).astype(float)

    def elementwise_null_score(
        self,
        X_train: ndarray,
        y_train: ndarray,
        X_test: ndarray,
        y_test: ndarray,
    ) -> ndarray:
        min_score = np.inf
        result = np.zeros_like(y_test)
        for x in np.unique(y_train):
            y_spoof = np.full_like(y_test, x, dtype=y_test.dtype)
            elementwise_sore = np.array(y_test == y_spoof, dtype=float)
            score = np.mean(elementwise_sore)
            if min_score > score:
                min_score = score
                result = elementwise_sore

        return result


def compute_groupings(X: ndarray, sensitive_features: Union[int, Sequence[int]]) -> ndarray:
    if not isinstance(sensitive_features, collections.abc.Sequence):
        sensitive_features = [sensitive_features]
    X_sf = X[:, sensitive_features]
    unique_values = np.unique(X_sf, axis=0)
    groupings = np.zeros(X.shape[0], dtype=int)
    for i, unique in enumerate(unique_values):
        idx = (X_sf == unique).all(axis=1).nonzero()
        groupings[idx] = i
    return groupings


def compute_tpr_and_fpr(y_test: ndarray, y_pred: ndarray, *, groupings: ndarray) -> Tuple[ndarray, ndarray]:
    groups = sorted(np.unique(groupings))
    n_groups = len(groups)
    tpr = np.zeros(n_groups, dtype=float)
    fpr = np.zeros(n_groups, dtype=float)

    for g in range(n_groups):
        idx_g = groupings == g
        y_test_g = y_test[idx_g]
        y_pred_g = y_pred[idx_g]
        y_eq = y_test_g == y_pred_g
        y_neq = y_test_g != y_pred_g
        y_pos = y_pred_g == 1
        y_neg = y_pred_g == 0
        tp = np.sum(y_eq & y_pos)
        tn = np.sum(y_eq & y_neg)
        fp = np.sum(y_neq & y_pos)
        fn = np.sum(y_neq & y_neg)
        fpr[g] = fp / (tn + fp) if fp > 0.0 else 0.0
        tpr[g] = tp / (tp + fn) if tp > 0.0 else 0.0

    return tpr, fpr


def equalized_odds_diff(y_test: ndarray, y_pred: ndarray, *, groupings: ndarray) -> float:

    tpr, fpr = compute_tpr_and_fpr(y_test, y_pred, groupings=groupings)

    tpr_d = np.max(tpr) - np.min(tpr)
    fpr_d = np.max(fpr) - np.min(fpr)
    return max(tpr_d, fpr_d)


class SklearnModelEqualizedOddsDifference(SklearnModelUtility):
    def __init__(
        self,
        model: SklearnModelOrPipeline,
        sensitive_features: Union[int, Sequence[int]],
        groupings: Optional[ndarray] = None,
    ) -> None:
        super().__init__(model, None)
        if not isinstance(sensitive_features, collections.abc.Sequence):
            sensitive_features = [sensitive_features]
        self._sensitive_features = sensitive_features
        self._groupings = groupings

    def _metric_score(
        self,
        metric: Optional[MetricCallable],
        y_test: ndarray,
        y_pred: ndarray,
        *,
        X_test: Optional[ndarray] = None,
        indices: Optional[ndarray] = None,
    ) -> float:
        assert X_test is not None
        if list(sorted(np.unique(y_test))) != [0, 1]:
            raise ValueError("This utility works only on binary classification problems.")
        groupings = self._groupings
        if groupings is None:
            groupings = compute_groupings(X_test, self._sensitive_features)
        elif indices is not None:
            groupings = groupings[indices]
            assert isinstance(groupings, ndarray)
        return equalized_odds_diff(y_test, y_pred, groupings=groupings)

    def elementwise_score(
        self,
        X_train: ndarray,
        y_train: ndarray,
        X_test: ndarray,
        y_test: ndarray,
    ) -> ndarray:
        if list(sorted(np.unique(y_test))) != [0, 1]:
            raise ValueError("This utility works only on binary classification problems.")

        n_train, n_test = X_train.shape[0], X_test.shape[0]
        utilities = np.zeros((n_train, n_test), dtype=float)

        try:
            # Precompute the true positive rate and false positive rate by using the entire training dataset.
            groupings = (
                self._groupings if self._groupings is not None else compute_groupings(X_test, self._sensitive_features)
            )
            model = self._model_fit(self.model, X_train, y_train)
            y_pred = self._model_predict(model, X_test)
            tpr, fpr = compute_tpr_and_fpr(y_test, y_pred, groupings=groupings)

            utilities_eq = np.equal.outer(y_train, y_test).astype(float)
            utilities_neq = 1 - utilities_eq
            idx_y_pos = y_test == 1
            idx_y_neg = y_test == 0
            if np.max(tpr) - np.min(tpr) > np.max(fpr) - np.min(fpr):
                g_max, g_min = np.argmax(tpr), np.argmin(tpr)
                idx_g_max = groupings == g_max
                idx_g_min = groupings == g_min
                idx_max = idx_g_max & idx_y_pos
                idx_min = idx_g_min & idx_y_pos
                utilities[:, idx_max] = utilities_eq[:, idx_max] / np.sum(idx_max)
                utilities[:, idx_min] = -utilities_eq[:, idx_min] / np.sum(idx_min)
            else:
                g_max, g_min = np.argmax(fpr), np.argmin(fpr)
                idx_g_max = groupings == g_max
                idx_g_min = groupings == g_min
                idx_max = idx_g_max & idx_y_neg
                idx_min = idx_g_min & idx_y_neg
                utilities[:, idx_max] = utilities_neq[:, idx_max] / np.sum(idx_max)
                utilities[:, idx_min] = -utilities_neq[:, idx_min] / np.sum(idx_min)
        except ValueError:
            pass

        return utilities

    def elementwise_null_score(
        self,
        X_train: ndarray,
        y_train: ndarray,
        X_test: ndarray,
        y_test: ndarray,
    ) -> ndarray:
        return np.zeros_like(y_test, dtype=float)


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
        provenance.shape[1] == 1
        and provenance.shape[3] == 1
        and provenance.shape[0] == provenance.shape[2]
        and np.all(np.equal(provenance[:, 0, :, 0], np.eye(n_units)))
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
