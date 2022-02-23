from __future__ import absolute_import
from copy import deepcopy
from math import floor

import datasets
import numpy as np

from abc import ABC, abstractmethod
from enum import Enum
from numpy import ndarray
from sklearn.datasets import fetch_openml, fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from typing import Dict, Optional, Sequence, Tuple, Type


class DatasetId(str, Enum):
    FashionMNIST = "FashionMNIST"
    UCI = "UCI"
    TwentyNewsGroups = "TwentyNewsGroups"


class DatasetModality(str, Enum):
    TABULAR = "tabular"
    IMAGE = "image"
    TEXT = "text"


DEFAULT_TRAINSIZE = 1000
DEFAULT_VALSIZE = 100
# DEFAULT_TRAINSIZE = 100
# DEFAULT_VALSIZE = 20
DEFAULT_SEED = 1
DEFAULT_CLASSES = [0, 6]


class Dataset(ABC):

    datasets: Dict[str, Type["Dataset"]] = {}
    _dataset: Optional[str] = None
    _modality: DatasetModality

    def __init__(
        self, trainsize: int = DEFAULT_TRAINSIZE, valsize: int = DEFAULT_VALSIZE, seed: int = DEFAULT_SEED, **kwargs
    ) -> None:
        self._trainsize = trainsize
        self._valsize = valsize
        self._seed = seed
        self._loaded: bool = False
        self._X_train: Optional[ndarray] = None
        self._y_train: Optional[ndarray] = None
        self._X_val: Optional[ndarray] = None
        self._y_val: Optional[ndarray] = None

    def __init_subclass__(cls: Type["Dataset"], modality: DatasetModality, id: Optional[str] = None) -> None:
        cls._dataset = id if id is not None else cls.__name__
        cls._modality = modality
        Dataset.datasets[cls._dataset] = cls

    @abstractmethod
    def load(self) -> None:
        pass

    @property
    def modality(self) -> DatasetModality:
        return self._modality

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def trainsize(self) -> int:
        return self._trainsize

    @property
    def valsize(self) -> int:
        return self._valsize

    @property
    def X_train(self) -> ndarray:
        if self._X_train is None:
            raise ValueError("The dataset is not loaded yet.")
        return self._X_train

    @X_train.setter
    def X_train(self, value: ndarray):
        self._X_train = value

    @property
    def y_train(self) -> ndarray:
        if self._y_train is None:
            raise ValueError("The dataset is not loaded yet.")
        return self._y_train

    @y_train.setter
    def y_train(self, value: ndarray):
        self._y_train = value

    @property
    def X_val(self) -> ndarray:
        if self._X_val is None:
            raise ValueError("The dataset is not loaded yet.")
        return self._X_val

    @X_val.setter
    def X_val(self, value: ndarray):
        self._X_val = value

    @property
    def y_val(self) -> ndarray:
        if self._y_val is None:
            raise ValueError("The dataset is not loaded yet.")
        return self._y_val

    @y_val.setter
    def y_val(self, value: ndarray):
        self._y_val = value

    def apply(self, pipeline: Pipeline) -> "Dataset":
        result = deepcopy(self)
        pipeline = deepcopy(pipeline)
        result._X_train = pipeline.fit_transform(result._X_train, result._y_train)
        result._X_val = pipeline.transform(result._X_val)
        return result


class BiasedMixin:
    @abstractmethod
    def load_biased(self, train_bias: float, val_bias: float = 0.0) -> None:
        pass

    @property
    def sensitive_feature(self) -> int:
        raise NotImplementedError()

    @property
    def train_bias(self) -> float:
        raise NotImplementedError()

    @property
    def val_bias(self) -> float:
        raise NotImplementedError()


def balance_train_and_valsize(trainsize: int, valsize: int, totsize: int) -> Tuple[int, int, int]:
    if trainsize > 0:
        if valsize > 0:
            if trainsize + valsize <= totsize:
                totsize = trainsize + valsize
            else:
                trainsize = round(totsize * trainsize / (trainsize + valsize))
                valsize = totsize - trainsize
        else:
            if trainsize >= totsize:
                raise ValueError("Requested trainsize %d more than available data %d." % (trainsize, totsize))
            else:
                valsize = totsize - trainsize
    else:
        if valsize > 0:
            if valsize >= totsize:
                raise ValueError("Requested valsize %d more than available data %d." % (valsize, totsize))
            else:
                trainsize = totsize - valsize
        else:
            trainsize = round(0.75 * totsize)
            valsize = totsize - trainsize
    return trainsize, valsize, totsize


def reduce_train_and_valsize(
    trainsize: int, valsize: int, totsize: int, requested_size: int, available_size: int
) -> Tuple[int, int, int]:
    if requested_size > available_size:
        reduction = float(available_size) / requested_size
        trainsize = floor(trainsize * reduction)
        valsize = floor(valsize * reduction)
        totsize = trainsize + valsize
    return trainsize, valsize, totsize


UCI_DEFAULT_SENSITIVE_FEATURE = 9


def compute_bias(X: ndarray, y: ndarray, sf: int) -> float:
    n_f0_l0 = np.sum((X[:, sf] == 0) & (y == 0))
    n_f0_l1 = np.sum((X[:, sf] == 0) & (y == 1))
    n_f1_l0 = np.sum((X[:, sf] == 1) & (y == 0))
    n_f1_l1 = np.sum((X[:, sf] == 1) & (y == 1))
    bias = float(n_f0_l0 + n_f1_l1 - n_f0_l1 - n_f1_l0) / (n_f0_l0 + n_f1_l1 + n_f0_l1 + n_f1_l0)
    return bias


class UCI(Dataset, BiasedMixin, modality=DatasetModality.TABULAR):
    def __init__(
        self,
        trainsize: int = DEFAULT_TRAINSIZE,
        valsize: int = DEFAULT_VALSIZE,
        seed: int = DEFAULT_SEED,
        sensitive_feature: int = UCI_DEFAULT_SENSITIVE_FEATURE,
        **kwargs
    ) -> None:
        super().__init__(trainsize, valsize, seed, **kwargs)
        self._sensitive_feature = sensitive_feature

    @property
    def sensitive_feature(self) -> int:
        return self._sensitive_feature

    @property
    def train_bias(self) -> float:
        if self._X_train is None or self._y_train is None:
            raise ValueError("Dataset not loaded yet.")
        return compute_bias(self._X_train, self._y_train, self._sensitive_feature)

    @property
    def val_bias(self) -> float:
        if self._X_val is None or self._y_val is None:
            raise ValueError("Dataset not loaded yet.")
        return compute_bias(self._X_val, self._y_val, self._sensitive_feature)

    def load(self) -> None:
        data = fetch_openml(data_id=1590, as_frame=False)
        X = np.nan_to_num(data.data)  # TODO: Maybe leave nan values.
        y = np.array(data.target == ">50K", dtype=int)
        trainsize = self.trainsize if self.trainsize > 0 else None
        valsize = self.valsize if self.valsize > 0 else None
        self._X_train, self._X_val, self._y_train, self._y_val = train_test_split(
            X, y, train_size=trainsize, test_size=valsize, random_state=self._seed
        )
        self._loaded = True
        assert self._X_train is not None and self._X_val is not None
        self._trainsize = self._X_train.shape[0]
        self._valsize = self._X_val.shape[0]

    def load_biased(
        self,
        train_bias: float,
        val_bias: float = 0.0,
    ) -> None:
        assert train_bias > -1.0 and train_bias < 1.0
        assert val_bias > -1.0 and val_bias < 1.0
        data = fetch_openml(data_id=1590, as_frame=False)
        X = np.nan_to_num(data.data)  # TODO: Maybe leave nan values.
        y = np.array(data.target == ">50K", dtype=int)

        n = X.shape[0]
        sf = self._sensitive_feature
        if not list(sorted(np.unique(X[:, sf]))) == [0, 1]:
            raise ValueError("The specified sensitive feature must be a binary feature.")
        train_bias = train_bias * 0.5 + 0.5
        val_bias = val_bias * 0.5 + 0.5
        idx_f0_l0 = np.nonzero((X[:, sf] == 0) & (y == 0))[0]
        idx_f0_l1 = np.nonzero((X[:, sf] == 0) & (y == 1))[0]
        idx_f1_l0 = np.nonzero((X[:, sf] == 1) & (y == 0))[0]
        idx_f1_l1 = np.nonzero((X[:, sf] == 1) & (y == 1))[0]
        indices = [idx_f0_l0, idx_f0_l1, idx_f1_l0, idx_f1_l1]

        trainsize, valsize, totsize = balance_train_and_valsize(self.trainsize, self.valsize, n)
        available_sizes = [idx.shape[0] for idx in indices]
        requested_sizes = [
            (0.5 * train_bias, 0.5 * val_bias),
            (0.5 * (1 - train_bias), 0.5 * (1 - val_bias)),
            (0.5 * (1 - train_bias), 0.5 * (1 - val_bias)),
            (0.5 * train_bias, 0.5 * val_bias),
        ]
        for available_size, (requested_trainsize, requested_valsize) in zip(available_sizes, requested_sizes):
            requested_size = floor(requested_trainsize * trainsize) + floor(requested_valsize * valsize)
            trainsize, valsize, totsize = reduce_train_and_valsize(
                trainsize, valsize, totsize, requested_size, available_size
            )

        random = np.random.RandomState(seed=self._seed)
        idx_train, idx_val = np.array([], dtype=int), np.array([], dtype=int)
        for i, idx in enumerate(indices):
            random.shuffle(idx)
            if i < 3:
                trainslice = floor(requested_sizes[i][0] * trainsize)
                valslice = floor(requested_sizes[i][1] * valsize)
            else:
                trainslice = trainsize - idx_train.shape[0]
                valslice = valsize - idx_val.shape[0]
            totslice = trainslice + valslice
            idx_train = np.concatenate((idx_train, idx[:trainslice]))
            idx_val = np.concatenate((idx_val, idx[trainslice:totslice]))
        random.shuffle(idx_train)
        random.shuffle(idx_val)

        self._X_train, self._y_train = X[idx_train], y[idx_train]
        self._X_val, self._y_val = X[idx_val], y[idx_val]
        assert self._X_train is not None and self._X_val is not None
        self._trainsize = self._X_train.shape[0]
        self._valsize = self._X_val.shape[0]


class FashionMNIST(Dataset, modality=DatasetModality.IMAGE):
    def __init__(
        self,
        trainsize: int = DEFAULT_TRAINSIZE,
        valsize: int = DEFAULT_VALSIZE,
        seed: int = DEFAULT_SEED,
        classes: Sequence[int] = DEFAULT_CLASSES,
        **kwargs
    ) -> None:
        self._classes = classes
        super().__init__(trainsize, valsize, seed, **kwargs)

    def classes(self) -> Sequence[int]:
        return self._classes

    def load(self) -> None:
        data = datasets.load_dataset("fashion_mnist")
        assert isinstance(data, datasets.dataset_dict.DatasetDict)

        # Select training and test validation examples based on the provided classes.
        train, val = data["train"], data["test"]
        assert isinstance(train, datasets.arrow_dataset.Dataset) and isinstance(val, datasets.arrow_dataset.Dataset)
        train_idx = np.where(np.isin(np.array(train["label"]), self._classes))[0]
        val_idx = np.where(np.isin(np.array(val["label"]), self._classes))[0]

        # Produce random samples of the training and validation sets based on the provided set sizes.
        random = np.random.RandomState(seed=self._seed)
        if self.trainsize > 0:
            train_idx = random.choice(train_idx, size=self.trainsize, replace=False)
        if self.valsize > 0:
            val_idx = random.choice(val_idx, size=self.valsize, replace=False)
        train_subset = train.select(train_idx)
        val_subset = val.select(val_idx)

        # Extract features.
        self._X_train = np.stack(train_subset["image"])
        self._X_val = np.stack(val_subset["image"])
        # TODO: Handle reshaping.

        # Encode labels.
        encoder = LabelEncoder()
        self._y_train = encoder.fit_transform(np.array(train_subset["label"], dtype=int))
        self._y_val = encoder.transform(np.array(val_subset["label"], dtype=int))

        self._loaded = True
        assert self._X_train is not None and self._X_val is not None
        self._trainsize = self._X_train.shape[0]
        self._valsize = self._X_val.shape[0]


class TwentyNewsGroups(Dataset, modality=DatasetModality.TEXT):
    def load(self) -> None:
        categories = ["comp.graphics", "sci.med"]
        train = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=self._seed)
        val = fetch_20newsgroups(subset="test", categories=categories, shuffle=True, random_state=self._seed)

        self._X_train, self._y_train = np.array(train.data), np.array(train.target)
        if self.trainsize > 0:
            self._X_train, self._y_train = self._X_train[: self.trainsize], self._y_train[: self.trainsize]
        self._X_val, self._y_val = np.array(val.data), np.array(val.target)
        if self.valsize > 0:
            self._X_val, self._y_val = self._X_val[: self.valsize], self._y_val[: self.valsize]
        self._loaded = True
        assert self._X_train is not None and self._X_val is not None
        self._trainsize = self._X_train.shape[0]
        self._valsize = self._X_val.shape[0]
