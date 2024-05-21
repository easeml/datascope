from __future__ import annotations

import numpy as np

from abc import ABC, abstractmethod
from numpy import ndarray
from numpy.typing import NDArray
from pandas import DataFrame, Series, Index, MultiIndex
from typing import Optional, Callable, Tuple, List, Hashable, Union, Protocol


class SklearnModel(Protocol):
    classes_: List[Hashable]

    def fit(
        self, X: Union[NDArray, DataFrame], y: Union[NDArray, Series], sample_weight: Optional[NDArray] = None
    ) -> None:
        pass

    def predict(self, X: Union[NDArray, DataFrame]) -> NDArray:
        pass

    def predict_proba(self, X: Union[NDArray, DataFrame]) -> NDArray:
        pass


class SklearnTransformer(Protocol):
    def fit(
        self, X: Union[NDArray, DataFrame], y: Union[NDArray, Series], sample_weight: Optional[NDArray] = None
    ) -> None:
        pass

    def transform(self, X: Union[NDArray, DataFrame]) -> NDArray:
        pass


class SklearnPipeline(SklearnModel, SklearnTransformer):
    steps: List[Tuple[str, Union[SklearnModel, SklearnTransformer]]]


SklearnModelOrPipeline = Union[SklearnModel, SklearnPipeline]


MetricCallable = Callable[[NDArray, NDArray], float]
DistanceCallable = Callable[[NDArray, NDArray], NDArray]


class ExtendedModelMixin(ABC):

    @abstractmethod
    def fit_extended(
        self,
        X: Union[NDArray, DataFrame],
        y: Union[NDArray, Series],
        metadata: Optional[Union[NDArray, DataFrame]] = None,
        X_val: Optional[Union[NDArray, DataFrame]] = None,
        y_val: Optional[Union[NDArray, Series]] = None,
        metadata_val: Optional[Union[NDArray, DataFrame]] = None,
    ) -> "ExtendedModelMixin":
        pass

    @abstractmethod
    def predict_extended(
        self,
        X: Union[NDArray, DataFrame],
        metadata: Optional[Union[NDArray, DataFrame]] = None,
    ) -> Union[NDArray, Series]:
        pass

    @abstractmethod
    def predict_proba_extended(
        self,
        X: Union[NDArray, DataFrame],
        metadata: Optional[Union[NDArray, DataFrame]] = None,
    ) -> Union[NDArray, DataFrame]:
        pass


def one_hot_encode_probabilities(y: NDArray, classes: List[Hashable]) -> NDArray:
    return (y[:, None] == np.array(classes)).astype(int)


class Postprocessor(ABC):

    def __init__(self, require_probabilities: bool = False) -> None:
        super().__init__()
        self.require_probabilities = require_probabilities

    @abstractmethod
    def fit(
        self,
        X: Union[NDArray, DataFrame],
        y: Union[NDArray, Series],
        metadata: Optional[Union[NDArray, DataFrame]] = None,
    ) -> "Postprocessor":
        pass

    @abstractmethod
    def transform(
        self,
        X: Union[NDArray, DataFrame],
        y: Union[NDArray, Series, DataFrame],
        y_proba: Optional[Union[NDArray, Series, DataFrame]] = None,
        metadata: Optional[Union[NDArray, DataFrame]] = None,
        output_probabilities: bool = False,
    ) -> Union[NDArray, Series, DataFrame]:
        pass


class IdentityPostprocessor(Postprocessor):
    _classes: Optional[List[Hashable]] = None

    def fit(
        self,
        X: Union[NDArray, DataFrame],
        y: Union[NDArray, Series],
        metadata: Optional[Union[NDArray, DataFrame]] = None,
    ) -> "Postprocessor":
        self._classes = np.unique(y).tolist()
        return self

    def transform(
        self,
        X: Union[NDArray, DataFrame],
        y: Union[NDArray, Series, DataFrame],
        y_proba: Optional[Union[NDArray, Series, DataFrame]] = None,
        metadata: Optional[Union[NDArray, DataFrame]] = None,
        output_probabilities: bool = False,
    ) -> Union[NDArray, Series, DataFrame]:
        if self._classes is None:
            raise ValueError("The postprocessor has not been fit.")
        if output_probabilities:
            if y_proba is None:
                if isinstance(y, Series) or isinstance(y, DataFrame):
                    y = y.to_numpy()
                assert isinstance(y, ndarray)
                y_proba = one_hot_encode_probabilities(y, self._classes)
            return y_proba
        else:
            return y


def expand_series_based_on_index(series: Series, index: Index) -> Series:
    if isinstance(series.index, MultiIndex):
        if isinstance(index, MultiIndex):
            if series.index.nlevels > index.nlevels:
                raise ValueError("The provided index must have at least as many levels as the series index.")
            return series.reindex(index.droplevel(list(range(series.index.nlevels, index.nlevels))))
        if series.index.nlevels > 1:
            raise ValueError(
                "The provided series has a MultiIndex with multiple levels, but the provided index is not a MultiIndex."
            )
        return series.reindex(index)
    if isinstance(series.index, Index):
        if isinstance(index, MultiIndex):
            return series.reindex(index.droplevel(list(range(1, index.nlevels))))
        else:
            return series.reindex(index)
