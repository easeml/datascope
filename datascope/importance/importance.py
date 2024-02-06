from abc import abstractmethod
from numpy import ndarray
from numpy.typing import NDArray
from pandas import DataFrame
from typing import Optional, Iterable

from ..utility import Provenance


class Importance:
    # @abstractmethod
    # def __call__(
    #     self, trainset: DataFrame, testset: DataFrame, units: Optional[Iterable[int]] = None
    # ) -> Iterable[float]:
    #     raise NotImplementedError()

    @abstractmethod
    def _fit(
        self, X: NDArray, y: NDArray, metadata: Optional[NDArray | DataFrame], provenance: Provenance
    ) -> "Importance":
        raise NotImplementedError()

    def fit(
        self,
        X: NDArray | DataFrame,
        y: NDArray | DataFrame,
        metadata: Optional[NDArray | DataFrame] = None,
        provenance: Optional[Provenance | NDArray] = None,
    ) -> "Importance":
        if hasattr(X, "provenance") and provenance is None:
            provenance = getattr(X, "provenance")
        if isinstance(provenance, ndarray):
            provenance = Provenance(data=provenance)
        elif provenance is None:
            provenance = Provenance(units=len(X))
        if isinstance(X, DataFrame):
            X = X.values
        if isinstance(y, DataFrame):
            y = y.values
        return self._fit(X, y, metadata, provenance)

    @abstractmethod
    def _score(
        self, X: NDArray, y: Optional[NDArray] = None, metadata: Optional[NDArray | DataFrame] = None, **kwargs
    ) -> Iterable[float]:
        raise NotImplementedError()

    def score(
        self,
        X: NDArray | DataFrame,
        y: Optional[NDArray | DataFrame] = None,
        metadata: Optional[NDArray | DataFrame] = None,
        **kwargs
    ) -> Iterable[float]:
        if isinstance(X, DataFrame):
            X = X.values
        if isinstance(y, DataFrame):
            y = y.values
        return self._score(X, y, metadata, **kwargs)
