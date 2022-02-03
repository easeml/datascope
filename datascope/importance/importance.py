from abc import abstractmethod
from numpy import ndarray
from pandas import DataFrame
from typing import Optional, Iterable, Union


class Importance:
    # @abstractmethod
    # def __call__(
    #     self, trainset: DataFrame, testset: DataFrame, units: Optional[Iterable[int]] = None
    # ) -> Iterable[float]:
    #     raise NotImplementedError()

    @abstractmethod
    def _fit(self, X: ndarray, y: ndarray, provenance: Optional[ndarray] = None) -> "Importance":
        raise NotImplementedError()

    def fit(
        self, X: Union[DataFrame, ndarray], y: Union[DataFrame, ndarray], provenance: Optional[ndarray] = None
    ) -> "Importance":
        if hasattr(X, "provenance") and provenance is None:
            provenance = getattr(X, "provenance")
        if isinstance(X, DataFrame):
            X = X.values
        if isinstance(y, DataFrame):
            y = y.values
        return self._fit(X, y, provenance)

    @abstractmethod
    def _score(self, X: ndarray, y: Optional[ndarray] = None, **kwargs) -> Iterable[float]:
        raise NotImplementedError()

    def score(
        self, X: Union[DataFrame, ndarray], y: Optional[Union[DataFrame, ndarray]] = None, **kwargs
    ) -> Iterable[float]:
        if isinstance(X, DataFrame):
            X = X.values
        if isinstance(y, DataFrame):
            y = y.values
        return self._score(X, y, **kwargs)
