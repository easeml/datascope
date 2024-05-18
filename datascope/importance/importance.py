from abc import abstractmethod
from numpy import ndarray
from numpy.typing import NDArray
from pandas import DataFrame, Series
from typing import Optional, Iterable, Union

from .common import expand_series_based_on_index
from ..utility import Provenance


class Importance:
    # @abstractmethod
    # def __call__(
    #     self, trainset: DataFrame, testset: DataFrame, units: Optional[Iterable[int]] = None
    # ) -> Iterable[float]:
    #     raise NotImplementedError()

    @abstractmethod
    def _fit(
        self, X: NDArray, y: NDArray, metadata: Optional[Union[NDArray, DataFrame]], provenance: Provenance
    ) -> "Importance":
        raise NotImplementedError()

    def fit(
        self,
        X: Union[NDArray, DataFrame],
        y: Union[NDArray, Series],
        metadata: Optional[Union[NDArray, DataFrame]] = None,
        provenance: Optional[Union[Provenance, NDArray]] = None,
    ) -> "Importance":
        if hasattr(X, "provenance") and provenance is None:
            provenance = getattr(X, "provenance")
        if isinstance(provenance, ndarray):
            provenance = Provenance(data=provenance)
        elif provenance is None:
            provenance = Provenance(units=len(X))
        if isinstance(y, Series):
            if len(y) != len(X):
                if isinstance(X, DataFrame):
                    y = expand_series_based_on_index(y, X.index)
                elif metadata is not None and isinstance(metadata, DataFrame):
                    y = expand_series_based_on_index(y, metadata.index)
                y = y.dropna()
                if len(y) != len(X):
                    raise ValueError("Length of y is not equal to X, even after reindexing.")
            y = y.to_numpy()
        if isinstance(X, DataFrame):
            X = X.values

        assert isinstance(y, ndarray)
        return self._fit(X, y, metadata, provenance)

    @abstractmethod
    def _score(
        self, X: NDArray, y: Optional[NDArray] = None, metadata: Optional[Union[NDArray, DataFrame]] = None, **kwargs
    ) -> Iterable[float]:
        raise NotImplementedError()

    def score(
        self,
        X: Union[NDArray, DataFrame],
        y: Optional[Union[NDArray, Series]] = None,
        metadata: Optional[Union[NDArray, DataFrame]] = None,
        **kwargs
    ) -> Iterable[float]:
        if isinstance(y, Series):
            if len(y) != len(X):
                if isinstance(X, DataFrame):
                    y = expand_series_based_on_index(y, X.index)
                elif metadata is not None and isinstance(metadata, DataFrame):
                    y = expand_series_based_on_index(y, metadata.index)
                y = y.dropna()
                if len(y) != len(X):
                    raise ValueError("Length of y is not equal to X, even after reindexing.")
            y = y.to_numpy()
        if isinstance(X, DataFrame):
            X = X.values

        assert isinstance(y, ndarray)
        return self._score(X, y, metadata, **kwargs)
