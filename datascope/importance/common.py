import collections.abc
import numpy as np
import warnings

from abc import ABC, abstractmethod
from copy import deepcopy
from collections import defaultdict
from functools import partial
from numpy import ndarray
from numpy.typing import NDArray
from pandas import DataFrame, Series
from typing import Iterable, Optional, Callable, Sequence, Tuple, List, Hashable
from typing_extensions import Protocol
from sklearn.base import clone
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder


class SklearnModel(Protocol):
    classes_: List[Hashable]

    def fit(self, X: NDArray | DataFrame, y: NDArray | Series, sample_weight: Optional[NDArray] = None) -> None:
        pass

    def predict(self, X: NDArray | DataFrame) -> NDArray:
        pass

    def predict_proba(self, X: NDArray | DataFrame) -> NDArray:
        pass


class SklearnTransformer(Protocol):
    def fit(self, X: NDArray | DataFrame, y: NDArray | Series, sample_weight: Optional[NDArray] = None) -> None:
        pass

    def transform(self, X: NDArray | DataFrame) -> NDArray:
        pass


class SklearnPipeline(SklearnModel, SklearnTransformer):
    steps: List[Tuple[str, SklearnModel | SklearnTransformer]]


SklearnModelOrPipeline = SklearnModel | SklearnPipeline


MetricCallable = Callable[[NDArray, NDArray], float]
DistanceCallable = Callable[[NDArray, NDArray], NDArray]


DEFAULT_SCORING_MAXITER = 100
DEFAULT_SEED = 7


class Postprocessor(ABC):

    def __init__(self, require_probabilities: bool = False) -> None:
        super().__init__()
        self.require_probabilities = require_probabilities

    @abstractmethod
    def fit(
        self, X: NDArray | DataFrame, y: NDArray | Series, metadata: Optional[NDArray | DataFrame] = None
    ) -> "Postprocessor":
        pass

    @abstractmethod
    def transform(
        self,
        X: NDArray | DataFrame,
        y: NDArray | Series,
        metadata: Optional[NDArray | DataFrame] = None,
        output_probabilities: bool = False,
    ) -> NDArray | Series:
        pass


class IdentityPostprocessor(Postprocessor):
    def fit(
        self, X: NDArray | DataFrame, y: NDArray | Series, metadata: Optional[NDArray | DataFrame] = None
    ) -> "Postprocessor":
        return self

    def transform(
        self,
        X: NDArray | DataFrame,
        y: NDArray | Series,
        metadata: Optional[NDArray | DataFrame] = None,
        output_probabilities: bool = False,
    ) -> NDArray | Series:
        return y


class Utility(ABC):
    @abstractmethod
    def __call__(
        self,
        X_train: NDArray | DataFrame,
        y_train: NDArray | Series,
        X_test: NDArray | DataFrame,
        y_test: NDArray | Series,
        metadata_train: Optional[NDArray | DataFrame] = None,
        metadata_test: Optional[NDArray | DataFrame] = None,
        null_score: Optional[float] = None,
        seed: int = DEFAULT_SEED,
    ) -> float:
        pass

    @abstractmethod
    def null_score(
        self,
        X_train: NDArray | DataFrame,
        y_train: NDArray | Series,
        X_test: NDArray | DataFrame,
        y_test: NDArray | Series,
    ) -> float:
        pass

    @abstractmethod
    def mean_score(
        self,
        X_train: NDArray | DataFrame,
        y_train: NDArray | Series,
        X_test: NDArray | DataFrame,
        y_test: NDArray | Series,
        maxiter: int = DEFAULT_SCORING_MAXITER,
        seed: int = DEFAULT_SEED,
    ) -> float:
        pass

    def elementwise_score(
        self,
        X_train: NDArray | DataFrame,
        y_train: NDArray | Series,
        X_test: NDArray | DataFrame,
        y_test: NDArray | Series,
    ) -> NDArray:
        raise NotImplementedError("This utility does not implement elementwise scoring.")

    def elementwise_null_score(
        self,
        X_train: NDArray | DataFrame,
        y_train: NDArray | Series,
        X_test: NDArray | DataFrame,
        y_test: NDArray | Series,
    ) -> NDArray:
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
        X_train: NDArray | DataFrame,
        y_train: NDArray | Series,
        X_test: NDArray | DataFrame,
        y_test: NDArray | Series,
        metadata_train: Optional[NDArray | DataFrame] = None,
        metadata_test: Optional[NDArray | DataFrame] = None,
        null_score: Optional[float] = None,
        seed: int = DEFAULT_SEED,
    ) -> float:
        scores = [
            u(
                X_train,
                y_train,
                X_test,
                y_test,
                metadata_train=metadata_train,
                metadata_test=metadata_test,
                null_score=np.nan,
                seed=seed,
            )
            for u in self._utilities
        ]
        if any(np.isnan(x) for x in scores):
            return null_score if null_score is not None else self.null_score(X_train, y_train, X_test, y_test)
        return sum(w * s for w, s in zip(self._weights, scores))

    def null_score(
        self,
        X_train: NDArray | DataFrame,
        y_train: NDArray | Series,
        X_test: NDArray | DataFrame,
        y_test: NDArray | Series,
    ) -> float:
        return sum(w * u.null_score(X_train, y_train, X_test, y_test) for w, u in zip(self._weights, self._utilities))

    def mean_score(
        self,
        X_train: NDArray | DataFrame,
        y_train: NDArray | Series,
        X_test: NDArray | DataFrame,
        y_test: NDArray | Series,
        maxiter: int = DEFAULT_SCORING_MAXITER,
        seed: int = DEFAULT_SEED,
    ) -> float:
        return sum(
            w * u.mean_score(X_train, y_train, X_test, y_test, maxiter=maxiter, seed=seed)
            for w, u in zip(self._weights, self._utilities)
        )

    def elementwise_score(
        self,
        X_train: NDArray | DataFrame,
        y_train: NDArray | Series,
        X_test: NDArray | DataFrame,
        y_test: NDArray | Series,
    ) -> NDArray:
        scores = np.stack(
            [w * u.elementwise_score(X_train, y_train, X_test, y_test) for w, u in zip(self._weights, self._utilities)]
        )
        return np.sum(scores, axis=0)

    def elementwise_null_score(
        self,
        X_train: NDArray | DataFrame,
        y_train: NDArray | Series,
        X_test: NDArray | DataFrame,
        y_test: NDArray | Series,
    ) -> NDArray:
        scores = np.stack(
            [
                w * u.elementwise_null_score(X_train, y_train, X_test, y_test)
                for w, u in zip(self._weights, self._utilities)
            ]
        )
        return np.sum(scores, axis=0)


class SklearnModelUtility(Utility):
    def __init__(
        self,
        model: SklearnModelOrPipeline,
        metric: Optional[MetricCallable],
        postprocessor: Optional[Postprocessor] = None,
        metric_requires_probabilities: bool = False,
    ) -> None:
        self.model = model
        self.metric = metric
        self.postprocessor = postprocessor if postprocessor is not None else IdentityPostprocessor()
        self.metric_requires_probabilities = metric_requires_probabilities

    def __call__(
        self,
        X_train: NDArray | DataFrame,
        y_train: NDArray | Series,
        X_test: NDArray | DataFrame,
        y_test: NDArray | Series,
        metadata_train: Optional[NDArray | DataFrame] = None,
        metadata_test: Optional[NDArray | DataFrame] = None,
        null_score: Optional[float] = None,
        seed: int = DEFAULT_SEED,
    ) -> float:
        score = 0.0
        if not isinstance(y_test, ndarray):
            y_test = y_test.to_numpy()
        with warnings.catch_warnings():
            warnings.simplefilter("error", category=RuntimeWarning)
            warnings.simplefilter("ignore", category=FutureWarning)
            try:
                # TODO: Ensure fit clears the model.
                np.random.seed(seed)
                model = self._model_fit(self.model, X_train, y_train)
                postprocessor = self._postprocessor_fit(self.postprocessor, X_train, y_train, metadata_train)
                y_pred = self._model_predict(
                    model,
                    X_test,
                    output_probabilities=self.metric_requires_probabilities or self.postprocessor.require_probabilities,
                )
                y_pred_processed = self._postprocessor_transform(postprocessor, X_test, y_pred, metadata_test)
                if self.metric_requires_probabilities and y_pred_processed.shape[1] == 2:
                    y_pred_processed = y_pred_processed[:, 1]
                score = self._metric_score(self.metric, y_test, y_pred_processed, X_test=X_test)
            except (ValueError, RuntimeWarning):
                try:
                    return null_score if null_score is not None else self.null_score(X_train, y_train, X_test, y_test)
                except (ValueError, RuntimeWarning):
                    return score
        return score

    def _model_fit(
        self, model: SklearnModelOrPipeline, X_train: NDArray | DataFrame, y_train: NDArray | Series
    ) -> SklearnModelOrPipeline:
        model = clone(model)
        model.fit(X_train, y_train)
        return model

    def _model_predict(
        self, model: SklearnModelOrPipeline, X_test: NDArray | DataFrame, output_probabilities: bool = False
    ) -> NDArray:
        if output_probabilities:
            if hasattr(model, "predict_proba"):
                return model.predict_proba(X_test)
            else:
                result = model.predict(X_test)
                return OneHotEncoder(categories=model.classes_).transform(result)
        else:
            return model.predict(X_test)

    def _postprocessor_fit(
        self,
        postprocessor: Postprocessor,
        X_train: NDArray | DataFrame,
        y_train: NDArray | Series,
        metadata_train: Optional[NDArray | DataFrame],
    ) -> Postprocessor:
        postprocessor = deepcopy(postprocessor)
        postprocessor.fit(X=X_train, y=y_train, metadata=metadata_train)
        return postprocessor

    def _postprocessor_transform(
        self,
        postprocessor: Postprocessor,
        X_test: NDArray | DataFrame,
        y_pred: NDArray | Series,
        metadata_test: Optional[NDArray | DataFrame],
        output_probabilities: bool = False,
    ) -> NDArray:
        if not isinstance(y_pred, ndarray):
            y_pred = y_pred.to_numpy()
        result = postprocessor.transform(
            X=X_test, y=y_pred, metadata=metadata_test, output_probabilities=output_probabilities
        )
        if not isinstance(result, ndarray):
            result = result.to_numpy()
        return result

    def _metric_score(
        self,
        metric: Optional[MetricCallable],
        y_test: NDArray,
        y_pred: NDArray,
        *,
        X_test: Optional[NDArray | DataFrame] = None,
        indices: Optional[NDArray] = None,
    ) -> float:
        if metric is None:
            raise ValueError("The metric was not provided.")
        return metric(y_test, y_pred)

    def null_score(
        self,
        X_train: NDArray | DataFrame,
        y_train: NDArray | Series,
        X_test: NDArray | DataFrame,
        y_test: NDArray | Series,
    ) -> float:
        scores = []
        if not isinstance(y_test, ndarray):
            y_test = y_test.to_numpy()
        for x in np.unique(y_train):
            y_spoof = np.full_like(y_test, x, dtype=y_test.dtype)
            scores.append(self._metric_score(self.metric, y_test, y_spoof, X_test=X_test))

        return min(scores)

    def mean_score(
        self,
        X_train: NDArray | DataFrame,
        y_train: NDArray | Series,
        X_test: NDArray | DataFrame,
        y_test: NDArray | Series,
        maxiter: int = DEFAULT_SCORING_MAXITER,
        seed: int = DEFAULT_SEED,
    ) -> float:
        np.random.seed(seed)
        model = self._model_fit(self.model, X_train, y_train)
        y_pred = self._model_predict(model, X_test)
        if not isinstance(y_test, ndarray):
            y_test = y_test.to_numpy()
        scores = []
        np.random.seed(seed)
        for _ in range(maxiter):
            idx = np.random.choice(len(y_test), int(len(y_test) * 0.5))
            X_test_subset: NDArray | DataFrame = X_test.iloc[idx] if isinstance(X_test, DataFrame) else X_test[idx, :]
            scores.append(self._metric_score(self.metric, y_test[idx], y_pred[idx], X_test=X_test_subset, indices=idx))
        return np.mean(scores).item()


class SklearnModelAccuracy(SklearnModelUtility):
    def __init__(self, model: SklearnModelOrPipeline, postprocessor: Optional[Postprocessor] = None) -> None:
        super().__init__(model, accuracy_score, postprocessor=postprocessor)

    def elementwise_score(
        self,
        X_train: NDArray | DataFrame,
        y_train: NDArray | Series,
        X_test: NDArray | DataFrame,
        y_test: NDArray | Series,
    ) -> NDArray:
        classes = np.unique(y_train)
        return np.equal.outer(classes, y_test).astype(float)

    def elementwise_null_score(
        self,
        X_train: NDArray | DataFrame,
        y_train: NDArray | Series,
        X_test: NDArray | DataFrame,
        y_test: NDArray | Series,
    ) -> NDArray:
        min_score = np.inf
        if not isinstance(y_test, ndarray):
            y_test = y_test.to_numpy()
        result = np.zeros_like(y_test)
        for x in np.unique(y_train):
            y_spoof = np.full_like(y_test, x, dtype=y_test.dtype)
            elementwise_sore = np.array(y_test == y_spoof, dtype=float)
            score = np.mean(elementwise_sore)
            if min_score > score:
                min_score = score
                result = elementwise_sore

        return result


class SklearnModelRocAuc(SklearnModelUtility):
    def __init__(self, model: SklearnModelOrPipeline, postprocessor: Optional[Postprocessor] = None) -> None:
        metric = partial(roc_auc_score, multi_class="ovr")  # We multi-class mode to be one-vs-rest.
        super().__init__(model, metric, postprocessor=postprocessor, metric_requires_probabilities=True)

    # def _model_predict(self, model: SklearnModelOrPipeline, X_test: NDArray | DataFrame) -> NDArray:
    #     y_test_pred_proba = model.predict_proba(X_test)
    #     if y_test_pred_proba.shape[1] == 2:
    #         y_test_pred_proba = y_test_pred_proba[:, 1]
    #     return y_test_pred_proba

    def null_score(
        self,
        X_train: NDArray | DataFrame,
        y_train: NDArray | Series,
        X_test: NDArray | DataFrame,
        y_test: NDArray | Series,
    ) -> float:
        scores = []
        classes = np.unique(y_train)
        if not isinstance(y_test, ndarray):
            y_test = y_test.to_numpy()
        for x in classes:
            y_spoof = np.full((y_test.shape[0], len(classes)), np.eye(len(classes))[x])
            if y_spoof.shape[1] == 2:
                y_spoof = y_spoof[:, 1]
            scores.append(self._metric_score(self.metric, y_test, y_spoof, X_test=X_test))

        return min(scores)

    def elementwise_score(
        self,
        X_train: NDArray | DataFrame,
        y_train: NDArray | Series,
        X_test: NDArray | DataFrame,
        y_test: NDArray | Series,
    ) -> NDArray:
        classes = np.unique(y_train)
        result = np.zeros((len(classes), len(y_test)))
        for c in classes:
            eq = np.equal.outer(classes, y_test)
            all_c = np.full_like(classes, c, dtype=classes.dtype)
            tp = (eq * np.equal.outer(all_c, y_test)).astype(float)
            tn = (eq * np.not_equal.outer(all_c, y_test)).astype(float)
            p = np.equal(y_test, c).sum(dtype=float)
            n = np.not_equal(y_test, c).sum(dtype=float)
            result += (tp / p + tn / n) * 0.5
        return result / float(len(classes))

    def elementwise_null_score(
        self,
        X_train: NDArray | DataFrame,
        y_train: NDArray | Series,
        X_test: NDArray | DataFrame,
        y_test: NDArray | Series,
    ) -> NDArray:
        if not isinstance(y_test, ndarray):
            y_test = y_test.to_numpy()
        result = np.zeros_like(y_test, dtype=float)
        classes, counts = np.unique(y_test, return_counts=True)
        least_frequent_class = classes[np.argmin(counts)]
        y_spoof = np.full_like(y_test, least_frequent_class, dtype=y_test.dtype)

        for c in classes:
            eq = np.equal(y_spoof, y_test)
            all_c = np.full_like(y_spoof, c, dtype=y_spoof.dtype)
            tp = (eq * np.equal(all_c, y_test)).astype(float)
            tn = (eq * np.not_equal(all_c, y_test)).astype(float)
            p = np.equal(y_test, c).sum(dtype=float)
            n = np.not_equal(y_test, c).sum(dtype=float)
            result += (tp / p + tn / n) * 0.5

        return result / float(len(classes))


def compute_groupings(
    X: NDArray | DataFrame, sensitive_features: int | Sequence[int] | str | Sequence[str]
) -> NDArray[np.int_]:
    if not isinstance(sensitive_features, collections.abc.Sequence):
        sensitive_features = [sensitive_features]
    X_sf: NDArray
    if isinstance(X, ndarray):
        if not all(isinstance(x, int) for x in sensitive_features):
            raise ValueError("The provided sensitive_features must be integers if the data is a numpy array.")
        X_sf = X[:, np.array(sensitive_features)]
    else:
        X_sf = X[sensitive_features].to_numpy()
    unique_values = np.unique(X_sf, axis=0)
    groupings: NDArray[np.int_] = np.zeros(X.shape[0], dtype=int)
    for i, unique in enumerate(unique_values):
        idx = (X_sf == unique).all(axis=1).nonzero()
        groupings[idx] = i
    return groupings


def compute_tpr_and_fpr(y_test: NDArray, y_pred: NDArray, *, groupings: NDArray[np.int_]) -> Tuple[NDArray, NDArray]:
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


def equalized_odds_diff(y_test: NDArray, y_pred: NDArray, *, groupings: NDArray[np.int_]) -> float:

    tpr, fpr = compute_tpr_and_fpr(y_test, y_pred, groupings=groupings)

    tpr_d = np.max(tpr) - np.min(tpr)
    fpr_d = np.max(fpr) - np.min(fpr)
    return max(tpr_d, fpr_d)


class SklearnModelEqualizedOddsDifference(SklearnModelUtility):
    def __init__(
        self,
        model: SklearnModelOrPipeline,
        sensitive_features: int | Sequence[int] | str | Sequence[str],
        groupings: Optional[NDArray] = None,
        postprocessor: Optional[Postprocessor] = None,
    ) -> None:
        super().__init__(model, None, postprocessor=postprocessor)
        if not isinstance(sensitive_features, collections.abc.Sequence):
            sensitive_features = [sensitive_features]
        self._sensitive_features = sensitive_features
        self._groupings = groupings

    def _metric_score(
        self,
        metric: Optional[MetricCallable],
        y_test: NDArray,
        y_pred: NDArray,
        *,
        X_test: Optional[NDArray | DataFrame] = None,
        indices: Optional[NDArray] = None,
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
        X_train: NDArray | DataFrame,
        y_train: NDArray | Series,
        X_test: NDArray | DataFrame,
        y_test: NDArray | Series,
    ) -> NDArray:
        if list(sorted(np.unique(y_test))) != [0, 1]:
            raise ValueError("This utility works only on binary classification problems.")

        n_test = X_test.shape[0]
        classes = np.unique(y_train)
        utilities = np.zeros((len(classes), n_test), dtype=float)
        if not isinstance(y_test, ndarray):
            y_test = y_test.to_numpy()

        try:
            # Precompute the true positive rate and false positive rate by using the entire training dataset.
            groupings = (
                self._groupings if self._groupings is not None else compute_groupings(X_test, self._sensitive_features)
            )
            model = self._model_fit(self.model, X_train, y_train)
            y_pred = self._model_predict(model, X_test)
            tpr, fpr = compute_tpr_and_fpr(y_test, y_pred, groupings=groupings)

            utilities_eq = np.equal.outer(classes, y_test).astype(float)
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
        X_train: NDArray | DataFrame,
        y_train: NDArray | Series,
        X_test: NDArray | DataFrame,
        y_test: NDArray | Series,
    ) -> NDArray:
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


def reshape(provenance: NDArray) -> NDArray:
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


def one_hot_encode(array: NDArray, mergelast: bool = False) -> NDArray:
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


def binarize(provenance: NDArray) -> NDArray:
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


def get_indices(provenance: NDArray, query: NDArray, simple_provenance: bool = False) -> NDArray:
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


class Provenance(NDArray):
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

    def get_indices(self, query: NDArray) -> NDArray:
        assert query.ndim in [1, 2]
        if query.ndim == 1:
            query = np.broadcast_to(query[:, np.newaxis], query.shape + (self.shape[-1],))
        query = np.broadcast_to(query, self.shape)
        land = np.logical_and(self, query)
        eq = np.equal(land, self)
        a1 = np.all(eq, axis=(2, 3))
        a2 = np.any(a1, axis=1)
        return a2

    def indices(self, units: Optional[NDArray] = None, world: Optional[NDArray | int] = None) -> NDArray:
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
        result: NDArray = np.pad(self, pad_width=pad_width)
        return result.view(Provenance)
