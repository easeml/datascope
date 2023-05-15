from abc import abstractmethod
from numpy import ndarray
from numpy.typing import NDArray
from pandas import DataFrame
from typing import Optional, Iterable, Union

from ..utility import Provenance


class Importance:
    # @abstractmethod
    # def __call__(
    #     self, trainset: DataFrame, testset: DataFrame, units: Optional[Iterable[int]] = None
    # ) -> Iterable[float]:
    #     raise NotImplementedError()

    @abstractmethod
    def _fit(self, X: NDArray, y: NDArray, provenance: Provenance) -> "Importance":
        raise NotImplementedError()

    def fit(
        self,
        X: Union[DataFrame, NDArray],
        y: Union[DataFrame, NDArray],
        provenance: Union[Provenance, NDArray, None] = None,
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
        return self._fit(X, y, provenance)

    @abstractmethod
    def _score(self, X: NDArray, y: Optional[NDArray] = None, **kwargs) -> Iterable[float]:
        raise NotImplementedError()

    def score(
        self, X: Union[DataFrame, NDArray], y: Optional[Union[DataFrame, NDArray]] = None, **kwargs
    ) -> Iterable[float]:
        if isinstance(X, DataFrame):
            X = X.values
        if isinstance(y, DataFrame):
            y = y.values
        return self._score(X, y, **kwargs)
