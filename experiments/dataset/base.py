from __future__ import absolute_import
import collections
from copy import deepcopy
from math import ceil, floor

import datasets
import numpy as np

from abc import ABC, abstractmethod
from enum import Enum
from numpy import ndarray
from sklearn.datasets import fetch_openml, fetch_20newsgroups, make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from typing import Dict, Optional, Sequence, Tuple, Type, Union

from datascope.importance.common import binarize, get_indices


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
DEFAULT_NUMFEATURES = 10
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
        self._provenance: Optional[ndarray] = None
        self._bin_provenance: Optional[ndarray] = None
        self._units: Optional[ndarray] = None

    def __init_subclass__(
        cls: Type["Dataset"], modality: Optional[DatasetModality] = None, id: Optional[str] = None
    ) -> None:
        if modality is None:
            return
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

    @property
    def provenance(self) -> ndarray:
        if self._provenance is None:
            raise ValueError("The dataset is not loaded yet.")
        return self._provenance

    @property
    def units(self) -> ndarray:
        if self._units is None:
            raise ValueError("The dataset is not loaded yet.")
        return self._units

    def _construct_provenance(self, groupings: Optional[ndarray] = None) -> None:
        if groupings is None:
            groupings = np.arange(self._trainsize, dtype=int)
        provenance = np.expand_dims(groupings, axis=(1, 2, 3))
        self._provenance = np.pad(provenance, pad_width=((0, 0), (0, 0), (0, 0), (0, 1)))
        self._units = np.sort(np.unique(groupings))

    def apply(self, pipeline: Pipeline) -> "Dataset":
        result = deepcopy(self)
        pipeline = deepcopy(pipeline)
        result._X_train = pipeline.fit_transform(result._X_train, result._y_train)
        result._X_val = pipeline.transform(result._X_val)
        return result

    # def corrupt_labels(self, probabilities: Union[float, Sequence[float]]) -> "Dataset":
    #     if not isinstance(probabilities, collections.Sequence):
    #         probabilities = [probabilities]
    #     result = deepcopy(self)
    #     n_groups = len(probabilities)
    #     n_examples = result.X_train.shape[0]
    #     n_examples_per_group = ceil(n_examples / float(n_groups))
    #     groupings = np.sort(np.tile(np.arange(n_groups), n_examples_per_group)[:n_examples])
    #     random = np.random.RandomState(seed=self._seed)
    #     for i, p in enumerate(probabilities):
    #         idx = groupings == i
    #         n_elements = np.sum(idx)
    #         idx[idx] = random.choice(a=[False, True], size=(n_elements), p=[1 - p, p])
    #         result._y_train[idx] = 1 - result.y_train[idx]
    #     result._provenance = result._construct_provenance(groupings=groupings)
    #     return result


class RandomDataset(Dataset, modality=DatasetModality.TABULAR, id="random"):
    def __init__(
        self,
        trainsize: int = DEFAULT_TRAINSIZE,
        valsize: int = DEFAULT_VALSIZE,
        seed: int = DEFAULT_SEED,
        numfeatures: int = DEFAULT_NUMFEATURES,
        **kwargs
    ) -> None:
        super().__init__(trainsize, valsize, seed, **kwargs)
        self._numfeatures = numfeatures

    @property
    def numfeatures(self) -> int:
        return self._numfeatures

    def load(self) -> None:

        X, y = make_classification(
            n_samples=self.trainsize + self.valsize,
            n_features=self.numfeatures,
            n_redundant=0,
            n_informative=self.numfeatures,
            n_repeated=0,
            n_classes=2,
            n_clusters_per_class=1,
            random_state=self._seed,
        )

        self._X_train, self._X_val, self._y_train, self._y_val = train_test_split(
            X, y, train_size=self.trainsize, test_size=self.valsize, random_state=self._seed
        )
        self._loaded = True
        assert self._X_train is not None and self._X_val is not None
        self._trainsize = self._X_train.shape[0]
        self._valsize = self._X_val.shape[0]
        self._construct_provenance()


class DirtyLabelDataset(Dataset):
    def __init__(
        self, trainsize: int = DEFAULT_TRAINSIZE, valsize: int = DEFAULT_VALSIZE, seed: int = DEFAULT_SEED, **kwargs
    ) -> None:
        super().__init__(trainsize, valsize, seed, **kwargs)
        self._y_train_dirty: Optional[ndarray] = None
        self._units_dirty: Optional[ndarray] = None
        self._groupings: Optional[ndarray] = None

    @property
    def units_dirty(self) -> ndarray:
        if self._X_train is None:
            raise ValueError("The dataset is not loaded yet.")
        if self._units_dirty is None:
            return np.zeros(self.trainsize, dtype=bool)
        else:
            return self._units_dirty

    @property
    def y_train(self) -> ndarray:
        y_train = super().y_train
        if self._y_train_dirty is not None:
            if self._bin_provenance is None:
                self._bin_provenance = binarize(self.provenance)
            idx_dirty = get_indices(self._bin_provenance, self.units_dirty)
            y_train = np.where(idx_dirty, self._y_train_dirty, y_train)
        return y_train

    @y_train.setter
    def y_train(self, value: ndarray):
        self._y_train = value

    def corrupt_labels(self, probabilities: Union[float, Sequence[float]]) -> "DirtyLabelDataset":
        if not self.loaded:
            raise ValueError("The dataset is not loaded yet.")
        result = deepcopy(self)
        assert result._y_train is not None
        result._y_train_dirty = deepcopy(result._y_train)
        n_examples = result.X_train.shape[0]
        if not isinstance(probabilities, collections.Sequence):
            probabilities = [probabilities for _ in range(n_examples)]
        n_groups = len(probabilities)
        n_examples_per_group = ceil(n_examples / float(n_groups))
        groupings = np.tile(np.arange(n_groups), n_examples_per_group)[:n_examples]
        random = np.random.RandomState(seed=self._seed)
        random.shuffle(groupings)
        dirty_idx = np.zeros(result.trainsize, dtype=bool)
        units_dirty = np.zeros(n_groups, dtype=bool)
        for i, p in enumerate(probabilities):
            idx: ndarray = groupings == i
            n_elements = np.sum(idx)
            idx[idx] = random.choice(a=[False, True], size=(n_elements), p=[1 - p, p])
            dirty_idx[idx] = True
            if idx.sum() > 0:
                units_dirty[i] = True
            result._y_train_dirty[idx] = 1 - result._y_train[idx]
        result._construct_provenance(groupings=groupings)
        result._groupings = groupings
        result._units_dirty = units_dirty
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


class UCI(DirtyLabelDataset, BiasedMixin, modality=DatasetModality.TABULAR):
    def __init__(
        self,
        trainsize: int = DEFAULT_TRAINSIZE,
        valsize: int = DEFAULT_VALSIZE,
        seed: int = DEFAULT_SEED,
        sensitive_feature: int = UCI_DEFAULT_SENSITIVE_FEATURE,
        **kwargs
    ) -> None:
        super().__init__(trainsize=trainsize, valsize=valsize, seed=seed, **kwargs)
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
        self._construct_provenance()

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
        self._construct_provenance()


class FashionMNIST(DirtyLabelDataset, modality=DatasetModality.IMAGE):
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
        self._construct_provenance()


class TwentyNewsGroups(DirtyLabelDataset, modality=DatasetModality.TEXT):
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
        self._construct_provenance()
