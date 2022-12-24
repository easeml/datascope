from __future__ import absolute_import

import collections
import datasets
import numpy as np
import os
import pandas as pd
import urllib.request

from abc import ABC, abstractmethod
from copy import deepcopy
from enum import Enum
from folktables import ACSDataSource, ACSIncome
from math import ceil, floor
from numpy import ndarray
from sklearn.datasets import fetch_openml, fetch_20newsgroups, make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from typing import Dict, List, Optional, Sequence, Tuple, Type, Union

from datascope.importance.common import binarize, get_indices
from datascope.importance.shapley import checknan


class DatasetId(str, Enum):
    FashionMNIST = "FashionMNIST"
    UCI = "UCI"
    TwentyNewsGroups = "TwentyNewsGroups"


class DatasetModality(str, Enum):
    TABULAR = "tabular"
    IMAGE = "image"
    TEXT = "text"


KEYWORD_REPLACEMENTS = {"UCI": "UCI Adult", "FolkUCI": "Folktables Adult", "TwentyNewsGroups": "20NewsGroups"}


DEFAULT_TRAINSIZE = 1000
DEFAULT_VALSIZE = 100
DEFAULT_TESTSIZE = 100
DEFAULT_NUMFEATURES = 10
# DEFAULT_TRAINSIZE = 100
# DEFAULT_VALSIZE = 20
DEFAULT_SEED = 1
DEFAULT_CLASSES = [0, 6]
DEFAULT_DATA_DIR = os.path.join("var", "data")


class Dataset(ABC):

    datasets: Dict[str, Type["Dataset"]] = {}
    _dataset: Optional[str] = None
    _modality: DatasetModality

    def __init__(
        self,
        trainsize: int = DEFAULT_TRAINSIZE,
        valsize: int = DEFAULT_VALSIZE,
        testsize: int = DEFAULT_TESTSIZE,
        seed: int = DEFAULT_SEED,
        **kwargs
    ) -> None:
        self._trainsize = trainsize
        self._valsize = valsize
        self._testsize = testsize
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

    @classmethod
    def preload(cls) -> None:
        pass

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
    def testsize(self) -> int:
        return self._testsize

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
    def X_test(self) -> ndarray:
        if self._X_test is None:
            raise ValueError("The dataset is not loaded yet.")
        return self._X_test

    @X_test.setter
    def X_test(self, value: ndarray):
        self._X_test = value

    @property
    def y_test(self) -> ndarray:
        if self._y_test is None:
            raise ValueError("The dataset is not loaded yet.")
        return self._y_test

    @y_test.setter
    def y_test(self, value: ndarray):
        self._y_test = value

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
            self._provenance = np.array(np.nan)
            self._units = np.arange(self._trainsize, dtype=int)
        else:
            provenance = np.expand_dims(groupings, axis=(1, 2, 3))
            self._provenance = np.pad(provenance, pad_width=((0, 0), (0, 0), (0, 0), (0, 1)))
            self._units = np.sort(np.unique(groupings))

    def apply(self, pipeline: Pipeline) -> "Dataset":
        result = deepcopy(self)
        pipeline = deepcopy(pipeline)
        result._X_train = pipeline.fit_transform(result._X_train, result._y_train)
        result._X_val = pipeline.transform(result._X_val)
        result._X_test = pipeline.transform(result._X_test)
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
        testsize: int = DEFAULT_TESTSIZE,
        seed: int = DEFAULT_SEED,
        numfeatures: int = DEFAULT_NUMFEATURES,
        **kwargs
    ) -> None:
        super().__init__(trainsize=trainsize, valsize=valsize, testsize=testsize, seed=seed, **kwargs)
        self._numfeatures = numfeatures

    @property
    def numfeatures(self) -> int:
        return self._numfeatures

    def load(self) -> None:

        X, y = make_classification(
            n_samples=self.trainsize + self.valsize + self.testsize,
            n_features=self.numfeatures,
            n_redundant=0,
            n_informative=self.numfeatures,
            n_repeated=0,
            n_classes=2,
            n_clusters_per_class=1,
            random_state=self._seed,
        )

        self._X_train, self._X_val, self._y_train, self._y_val = train_test_split(
            X, y, train_size=self.trainsize, test_size=self.valsize + self.testsize, random_state=self._seed
        )
        self._X_val, self._X_test, self._y_val, self._y_test = train_test_split(
            self._X_val, self._y_val, train_size=self.valsize, test_size=self.testsize, random_state=self._seed
        )
        self._loaded = True
        assert self._X_train is not None and self._X_val is not None
        self._trainsize = self._X_train.shape[0]
        self._valsize = self._X_val.shape[0]
        self._testsize = self._X_test.shape[0]
        self._construct_provenance()


class DirtyLabelDataset(Dataset):
    def __init__(
        self,
        trainsize: int = DEFAULT_TRAINSIZE,
        valsize: int = DEFAULT_VALSIZE,
        testsize: int = DEFAULT_TESTSIZE,
        seed: int = DEFAULT_SEED,
        **kwargs
    ) -> None:
        super().__init__(trainsize=trainsize, valsize=valsize, testsize=testsize, seed=seed, **kwargs)
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
            provenance = self.provenance
            if (
                provenance is None
                or checknan(provenance)
                or provenance.ndim == 4
                and np.all(provenance[:, 0, 0, 0] == np.arange(provenance.shape[0]))
            ):
                y_train = np.where(self.units_dirty, self._y_train_dirty, y_train)
            else:
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
        random = np.random.RandomState(seed=self._seed)
        if not isinstance(probabilities, collections.abc.Sequence):
            assert isinstance(probabilities, float)
            dirty_idx = random.choice(a=[False, True], size=(n_examples), p=[1 - probabilities, probabilities])
            result._y_train_dirty[dirty_idx] = 1 - result._y_train[dirty_idx]
            units_dirty = dirty_idx
            groupings = None
        else:
            n_groups = len(probabilities)
            n_examples_per_group = ceil(n_examples / float(n_groups))
            groupings = np.tile(np.arange(n_groups), n_examples_per_group)[:n_examples]
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


class BiasMethod(str, Enum):
    Feature = "feature"
    Label = "label"
    FeatureLabel = "featurelabel"


DEFAULT_BIAS_METHOD = BiasMethod.FeatureLabel


class BiasedMixin:
    @abstractmethod
    def load_biased(
        self,
        train_bias: float,
        val_bias: float = 0.0,
        test_bias: float = 0.0,
        bias_method: BiasMethod = DEFAULT_BIAS_METHOD,
    ) -> None:
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

    @staticmethod
    def _get_biased_indices(
        X: ndarray,
        y: ndarray,
        sensitive_feature: int,
        trainsize: int,
        valsize: int,
        testsize: int,
        train_bias: float,
        val_bias: float = 0.0,
        test_bias: float = 0.0,
        bias_method: BiasMethod = DEFAULT_BIAS_METHOD,
        seed: int = DEFAULT_SEED,
    ) -> Tuple[ndarray, ndarray, ndarray]:
        n = X.shape[0]
        sensitive_feature_values = list(sorted(np.unique(X[:, sensitive_feature])))
        if not len(sensitive_feature_values) == 2:
            raise ValueError("The specified sensitive feature must be a binary feature.")
        f0, f1 = sensitive_feature_values
        label_values = list(sorted(np.unique(y)))
        if not len(label_values) == 2:
            raise ValueError("The lebel must be a binary value.")
        l0, l1 = label_values
        train_bias = train_bias * 0.5 + 0.5
        val_bias = val_bias * 0.5 + 0.5
        test_bias = test_bias * 0.5 + 0.5

        # The biasing method will determine how we will stratify the data examples.
        idx_f0_l0 = np.nonzero((X[:, sensitive_feature] == f0) & (y == l0))[0]
        idx_f0_l1 = np.nonzero((X[:, sensitive_feature] == f0) & (y == l1))[0]
        idx_f1_l0 = np.nonzero((X[:, sensitive_feature] == f1) & (y == l0))[0]
        idx_f1_l1 = np.nonzero((X[:, sensitive_feature] == f1) & (y == l1))[0]
        if bias_method == BiasMethod.FeatureLabel:
            indices = [idx_f0_l0, idx_f0_l1, idx_f1_l0, idx_f1_l1]
            requested_portions = [
                (0.5 * train_bias, 0.5 * val_bias, 0.5 * test_bias),
                (0.5 * (1 - train_bias), 0.5 * (1 - val_bias), 0.5 * (1 - test_bias)),
                (0.5 * (1 - train_bias), 0.5 * (1 - val_bias), 0.5 * (1 - test_bias)),
                (0.5 * train_bias, 0.5 * val_bias, 0.5 * test_bias),
            ]
        elif bias_method == BiasMethod.Feature:
            indices = [np.concatenate((idx_f0_l0, idx_f0_l1)), np.concatenate((idx_f1_l0, idx_f1_l1))]
            requested_portions = [
                (train_bias, val_bias, test_bias),
                (1 - train_bias, 1 - val_bias, 1 - test_bias),
            ]
        elif bias_method == BiasMethod.Label:
            indices = [np.concatenate((idx_f0_l0, idx_f1_l0)), np.concatenate((idx_f0_l1, idx_f1_l1))]
            requested_portions = [
                (train_bias, val_bias, test_bias),
                (1 - train_bias, 1 - val_bias, 1 - test_bias),
            ]
        else:
            raise ValueError("Unknown bias method passed: %s" % repr(bias_method))

        trainsize, valsize, testsize, totsize = balance_train_val_and_testsize(trainsize, valsize, testsize, n)
        available_sizes = [idx.shape[0] for idx in indices]

        for available_size, (requested_trainportion, requested_valportion, requested_testportion) in zip(
            available_sizes, requested_portions
        ):
            requested_size = (
                floor(requested_trainportion * trainsize)
                + floor(requested_valportion * valsize)
                + floor(requested_testportion * testsize)
            )
            trainsize, valsize, testsize, totsize = reduce_train_val_and_testsize(
                trainsize, valsize, testsize, totsize, requested_size, available_size
            )

        random = np.random.RandomState(seed=seed)
        idx_train, idx_val, idx_test = np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int)
        for i, idx in enumerate(indices):
            random.shuffle(idx)
            if i < len(indices) - 1:
                trainslice = floor(requested_portions[i][0] * trainsize)
                valslice = floor(requested_portions[i][1] * valsize)
                testslice = floor(requested_portions[i][2] * testsize)
            else:
                trainslice = trainsize - idx_train.shape[0]
                valslice = valsize - idx_val.shape[0]
                testslice = testsize - idx_test.shape[0]
            trainvalslice = trainslice + valslice
            totslice = trainslice + valslice + testslice
            idx_train = np.concatenate((idx_train, idx[:trainslice]))
            idx_val = np.concatenate((idx_val, idx[trainslice:trainvalslice]))
            idx_test = np.concatenate((idx_test, idx[trainvalslice:totslice]))
        random.shuffle(idx_train)
        random.shuffle(idx_val)
        random.shuffle(idx_test)

        return idx_train, idx_val, idx_test


def reduce_train_val_and_testsize(
    trainsize: int, valsize: int, totsize: int, testsize: int, requested_size: int, available_size: int
) -> Tuple[int, int, int, int]:
    if requested_size > available_size:
        reduction = float(available_size) / requested_size
        trainsize = floor(trainsize * reduction)
        valsize = floor(valsize * reduction)
        testsize = floor(testsize * reduction)
        totsize = trainsize + valsize + testsize
    return trainsize, valsize, testsize, totsize


def balance_train_val_and_testsize(
    trainsize: int, valsize: int, testsize: int, totsize: int
) -> Tuple[int, int, int, int]:

    assert trainsize >= 0
    assert valsize >= 0
    assert testsize >= 0

    # If any of the sizes exceeds total available size, raise an exception.
    if trainsize > totsize:
        raise ValueError("Requested trainsize %d more than available data %d." % (trainsize, totsize))
    if valsize > totsize:
        raise ValueError("Requested valsize %d more than available data %d." % (valsize, totsize))
    if testsize > totsize:
        raise ValueError("Requested testsize %d more than available data %d." % (testsize, totsize))

    # If all sizes are set to zero then we distribute the total size according to fixed ratios.
    if trainsize == 0 and valsize == 0 and testsize == 0:
        trainsize = round(0.7 * totsize)
        valsize = round(0.15 * totsize)
        testsize = totsize - trainsize - valsize
    elif trainsize > 0 and valsize > 0 and testsize > 0:
        # If all sizes are set, then we just need to ensure that they do not take up more than the total available size.
        if trainsize + valsize + testsize > totsize:
            trainvaltestsize = trainsize + valsize + testsize
            trainsize = round(totsize * trainsize / trainvaltestsize)
            valsize = round(totsize * valsize / trainvaltestsize)
            testsize = totsize - trainsize - valsize
        else:
            totsize = trainsize + valsize + testsize
    else:
        # If some sizes are set and others are zero then we compute the
        # remainder and distribute it among the unset ones.
        fixedsize = trainsize + valsize + testsize
        remainder = totsize - fixedsize
        if remainder < 0:
            raise ValueError("Requested %d data examples, more than %d available data examples." % (fixedsize, totsize))

        # Establish ratios of unassigned sizes and normalize them.
        trainratio = 0.0 if trainsize > 0 else 0.7
        valratio = 0.0 if valsize > 0 else 0.15
        testratio = 0.0 if testsize > 0 else 0.15
        totratio = trainratio + valratio + testratio
        trainratio /= totratio
        valratio /= totratio
        testratio /= totratio

        # Use normalized ratios to distribute the remainder size.
        trainsize = trainsize if trainsize > 0 else round(remainder * trainratio)
        valsize = valsize if valsize > 0 else round(remainder * valratio)
        testsize = trainsize if trainsize > 0 else totsize - trainsize - valsize

    return trainsize, valsize, testsize, totsize


class BiasedDirtyLabelDataset(DirtyLabelDataset, BiasedMixin):
    def corrupt_labels_with_bias(
        self,
        probabilities: Union[float, Sequence[float]],
        groupbias: float = 0.0,
    ) -> "DirtyLabelDataset":
        if not self.loaded:
            raise ValueError("The dataset is not loaded yet.")
        result = deepcopy(self)
        assert result._y_train is not None
        result._y_train_dirty = deepcopy(result._y_train)
        n_examples = result.X_train.shape[0]

        # Determine indices of the sensitive feature groups.
        assert result._X_train is not None
        sensitive_feature_values = list(sorted(np.unique(result._X_train[:, self.sensitive_feature])))
        if not len(sensitive_feature_values) == 2:
            raise ValueError("The specified sensitive feature must be a binary feature.")
        f0, f1 = sensitive_feature_values
        idx_f0 = np.nonzero(result._X_train[:, self.sensitive_feature] == f0)[0]
        idx_f1 = np.nonzero(result._X_train[:, self.sensitive_feature] == f1)[0]
        groupbias = 0.5 * groupbias + 0.5
        r_f0 = float(len(idx_f0)) / float(len(idx_f0) + len(idx_f1))
        r_f1 = float(len(idx_f1)) / float(len(idx_f0) + len(idx_f1))

        random = np.random.RandomState(seed=self._seed)
        if not isinstance(probabilities, collections.abc.Sequence):
            assert isinstance(probabilities, float)
            p = probabilities
            p_f0, p_f1 = (p / r_f0) * groupbias, (p / r_f1) * (1 - groupbias)
            dirty_idx = np.zeros(shape=n_examples, dtype=bool)
            dirty_idx[idx_f0] = random.choice(a=[False, True], size=len(idx_f0), p=[1 - p_f0, p_f0])
            dirty_idx[idx_f1] = random.choice(a=[False, True], size=len(idx_f1), p=[1 - p_f1, p_f1])
            result._y_train_dirty[dirty_idx] = 1 - result._y_train[dirty_idx]
            units_dirty = dirty_idx
            groupings = None
        else:
            n_groups = len(probabilities)
            n_examples_per_group = ceil(n_examples / float(n_groups))
            groupings = np.tile(np.arange(n_groups), n_examples_per_group)[:n_examples]
            random.shuffle(groupings)
            dirty_idx = np.zeros(result.trainsize, dtype=bool)
            units_dirty = np.zeros(n_groups, dtype=bool)
            for i, p in enumerate(probabilities):
                idx_g_f0, idx_g_f1 = np.zeros(result.trainsize, dtype=bool), np.zeros(result.trainsize, dtype=bool)
                idx_g_f0[idx_f0] = groupings[idx_f0] == i
                idx_g_f1[idx_f1] = groupings[idx_f1] == i
                n_elements_f0 = np.sum(idx_g_f0)
                n_elements_f1 = np.sum(idx_g_f1)
                p_f0, p_f1 = (p / r_f0) * groupbias, (p / r_f1) * (1 - groupbias)
                p_f0, p_f1 = min(max(p_f0, 0.0), 1.0), min(max(p_f1, 0.0), 1.0)
                idx_g_f0[idx_g_f0] = random.choice(a=[False, True], size=(n_elements_f0), p=[1 - p_f0, p_f0])
                idx_g_f1[idx_g_f1] = random.choice(a=[False, True], size=(n_elements_f1), p=[1 - p_f1, p_f1])
                if idx_g_f0.sum() + idx_g_f1.sum() > 0:
                    units_dirty[i] = True
                result._y_train_dirty[idx_g_f0] = 1 - result._y_train[idx_g_f0]
                result._y_train_dirty[idx_g_f1] = 1 - result._y_train[idx_g_f1]
        result._construct_provenance(groupings=groupings)
        result._groupings = groupings
        result._units_dirty = units_dirty
        return result


UCI_DEFAULT_SENSITIVE_FEATURE = 9
FOLK_UCI_DEFAULT_SENSITIVE_FEATURE = 8


def compute_bias(X: ndarray, y: ndarray, sf: int) -> float:
    n_f0_l0 = np.sum((X[:, sf] == 0) & (y == 0))
    n_f0_l1 = np.sum((X[:, sf] == 0) & (y == 1))
    n_f1_l0 = np.sum((X[:, sf] == 1) & (y == 0))
    n_f1_l1 = np.sum((X[:, sf] == 1) & (y == 1))
    bias = float(n_f0_l0 + n_f1_l1 - n_f0_l1 - n_f1_l0) / (n_f0_l0 + n_f1_l1 + n_f0_l1 + n_f1_l0)
    return bias


class UCI(BiasedDirtyLabelDataset, modality=DatasetModality.TABULAR):
    def __init__(
        self,
        trainsize: int = DEFAULT_TRAINSIZE,
        valsize: int = DEFAULT_VALSIZE,
        testsize: int = DEFAULT_TESTSIZE,
        seed: int = DEFAULT_SEED,
        sensitive_feature: int = UCI_DEFAULT_SENSITIVE_FEATURE,
        **kwargs
    ) -> None:
        super().__init__(trainsize=trainsize, valsize=valsize, testsize=testsize, seed=seed, **kwargs)
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

    @property
    def test_bias(self) -> float:
        if self._X_test is None or self._y_test is None:
            raise ValueError("Dataset not loaded yet.")
        return compute_bias(self._X_test, self._y_test, self._sensitive_feature)

    @classmethod
    def preload(cls) -> None:
        fetch_openml(data_id=1590, as_frame=False, data_home=DEFAULT_DATA_DIR, parser="pandas")

    def load(self) -> None:
        data = fetch_openml(data_id=1590, as_frame=False, data_home=DEFAULT_DATA_DIR, parser="liac-arff")
        X = np.nan_to_num(data.data)  # TODO: Maybe leave nan values.
        y = np.array(data.target == ">50K", dtype=int)

        # # Make dataset labels balanced.
        # idx_l0 = np.nonzero(y == 0)[0]
        # idx_l1 = np.nonzero(y == 1)[0]
        # if len(idx_l0) > len(idx_l1):
        #     idx_l0 = idx_l0[: len(idx_l1)]
        # else:
        #     idx_l1 = idx_l1[: len(idx_l0)]
        # idx = np.concatenate([idx_l0, idx_l1])
        # X = X[idx]
        # y = y[idx]

        trainsize = self.trainsize if self.trainsize > 0 else None
        valsize = self.valsize if self.valsize > 0 else None
        testsize = self.testsize if self.testsize > 0 else None
        valtestsize = valsize + testsize if valsize is not None and testsize is not None else None
        self._X_train, self._X_val, self._y_train, self._y_val = train_test_split(
            X, y, train_size=trainsize, test_size=valtestsize, random_state=self._seed
        )
        self._X_val, self._X_test, self._y_val, self._y_test = train_test_split(
            self._X_val, self._y_val, train_size=valsize, test_size=testsize, random_state=self._seed
        )

        self._loaded = True
        assert self._X_train is not None and self._X_val is not None
        self._trainsize = self._X_train.shape[0]
        self._valsize = self._X_val.shape[0]
        self._testsize = self._X_test.shape[0]
        self._construct_provenance()

    def load_biased(
        self,
        train_bias: float,
        val_bias: float = 0.0,
        test_bias: float = 0.0,
        bias_method: BiasMethod = DEFAULT_BIAS_METHOD,
    ) -> None:
        assert train_bias > -1.0 and train_bias < 1.0
        assert val_bias > -1.0 and val_bias < 1.0
        data = fetch_openml(data_id=1590, as_frame=False, data_home=DEFAULT_DATA_DIR, parser="pandas")
        X = np.nan_to_num(data.data)  # TODO: Maybe leave nan values.
        y = np.array(data.target == ">50K", dtype=int)

        idx_train, idx_val, idx_test = BiasedMixin._get_biased_indices(
            X=X,
            y=y,
            sensitive_feature=self.sensitive_feature,
            trainsize=self.trainsize,
            valsize=self.valsize,
            testsize=self.testsize,
            train_bias=train_bias,
            val_bias=val_bias,
            test_bias=test_bias,
            bias_method=bias_method,
            seed=self._seed,
        )

        self._X_train, self._y_train = X[idx_train], y[idx_train]
        self._X_val, self._y_val = X[idx_val], y[idx_val]
        self._X_test, self._y_test = X[idx_test], y[idx_test]
        assert self._X_train is not None and self._X_val is not None and self._X_test is not None
        self._trainsize = self._X_train.shape[0]
        self._valsize = self._X_val.shape[0]
        self._testsize = self._X_test.shape[0]
        self._construct_provenance()


FOLKUCI_STATE_COUNTS_2018 = {
    "AL": 22268,
    "AK": 3546,
    "AZ": 33277,
    "AR": 13929,
    "CA": 195665,
    "CO": 31306,
    "CT": 19785,
    "DE": 4713,
    "FL": 98925,
    "GA": 50915,
    "HI": 7731,
    "ID": 8265,
    "IL": 67016,
    "IN": 35022,
    "IA": 17745,
    "KS": 15807,
    "KY": 22006,
    "LA": 20667,
    "ME": 7002,
    "MD": 33042,
    "MA": 40114,
    "MI": 50008,
    "MN": 31021,
    "MS": 13189,
    "MO": 31664,
    "MT": 5463,
    "NE": 10785,
    "NV": 14807,
    "NH": 7966,
    "NJ": 47781,
    "NM": 8711,
    "NY": 103021,
    "NC": 52067,
    "ND": 4455,
    "OH": 62135,
    "OK": 17917,
    "OR": 21919,
    "PA": 68308,
    "RI": 5712,
    "SC": 24879,
    "SD": 4899,
    "TN": 34003,
    "TX": 135924,
    "UT": 16337,
    "VT": 3767,
    "VA": 46144,
    "WA": 39944,
    "WV": 8103,
    "WI": 32690,
    "WY": 3064,
    "PR": 9071,
}


def get_states_for_size(n: int) -> List[str]:
    if n <= 0:
        return list(FOLKUCI_STATE_COUNTS_2018.keys())
    result: List[str] = []
    totsize = 0
    for state in sorted(FOLKUCI_STATE_COUNTS_2018, key=lambda x: FOLKUCI_STATE_COUNTS_2018[x], reverse=False):
        result.append(state)
        totsize += FOLKUCI_STATE_COUNTS_2018[state]
        if totsize > n:
            break
    return result


class FolkUCI(BiasedDirtyLabelDataset, modality=DatasetModality.TABULAR):
    def __init__(
        self,
        trainsize: int = DEFAULT_TRAINSIZE,
        valsize: int = DEFAULT_VALSIZE,
        testsize: int = DEFAULT_TESTSIZE,
        seed: int = DEFAULT_SEED,
        sensitive_feature: int = FOLK_UCI_DEFAULT_SENSITIVE_FEATURE,
        **kwargs
    ) -> None:
        super().__init__(trainsize=trainsize, valsize=valsize, testsize=testsize, seed=seed, **kwargs)
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

    @property
    def test_bias(self) -> float:
        if self._X_test is None or self._y_test is None:
            raise ValueError("Dataset not loaded yet.")
        return compute_bias(self._X_test, self._y_test, self._sensitive_feature)

    @classmethod
    def preload(cls) -> None:
        data_source = ACSDataSource(survey_year="2018", horizon="1-Year", survey="person", root_dir=DEFAULT_DATA_DIR)
        data_source.get_data(download=True)

    def load(self) -> None:
        data_source = ACSDataSource(survey_year="2018", horizon="1-Year", survey="person", root_dir=DEFAULT_DATA_DIR)
        n = (
            self.trainsize + self.valsize + self.testsize
            if self.trainsize != 0 and self.valsize != 0 and self.testsize != 0
            else 0
        )
        states = get_states_for_size(n)
        acs_data = data_source.get_data(states=states, download=True)
        X, y, _ = ACSIncome.df_to_numpy(acs_data)
        del acs_data

        trainsize = self.trainsize if self.trainsize > 0 else None
        valsize = self.valsize if self.valsize > 0 else None
        testsize = self.testsize if self.testsize > 0 else None
        valtestsize = valsize + testsize if valsize is not None and testsize is not None else None
        self._X_train, self._X_val, self._y_train, self._y_val = train_test_split(
            X, y, train_size=trainsize, test_size=valtestsize, random_state=self._seed
        )
        self._X_val, self._X_test, self._y_val, self._y_test = train_test_split(
            self._X_val, self._y_val, train_size=valsize, test_size=testsize, random_state=self._seed
        )
        del X, y

        self._loaded = True
        assert self._X_train is not None and self._X_val is not None
        self._trainsize = self._X_train.shape[0]
        self._valsize = self._X_val.shape[0]
        self._testsize = self._X_test.shape[0]
        self._construct_provenance()

    def load_biased(
        self,
        train_bias: float,
        val_bias: float = 0.0,
        test_bias: float = 0.0,
        bias_method: BiasMethod = DEFAULT_BIAS_METHOD,
    ) -> None:
        assert train_bias > -1.0 and train_bias < 1.0
        assert val_bias > -1.0 and val_bias < 1.0
        data_source = ACSDataSource(survey_year="2018", horizon="1-Year", survey="person", root_dir=DEFAULT_DATA_DIR)
        n = (
            self.trainsize + self.valsize + self.testsize
            if self.trainsize != 0 and self.valsize != 0 and self.testsize != 0
            else 0
        )
        states = get_states_for_size(n)
        acs_data = data_source.get_data(states=states, download=True)
        X, y, _ = ACSIncome.df_to_numpy(acs_data)
        del acs_data

        idx_train, idx_val, idx_test = BiasedMixin._get_biased_indices(
            X=X,
            y=y,
            sensitive_feature=self.sensitive_feature,
            trainsize=self.trainsize,
            valsize=self.valsize,
            testsize=self.testsize,
            train_bias=train_bias,
            val_bias=val_bias,
            test_bias=test_bias,
            bias_method=bias_method,
            seed=self._seed,
        )

        self._X_train, self._y_train = X[idx_train], y[idx_train]
        self._X_val, self._y_val = X[idx_val], y[idx_val]
        self._X_test, self._y_test = X[idx_test], y[idx_test]
        assert self._X_train is not None and self._X_val is not None and self._X_test is not None
        self._trainsize = self._X_train.shape[0]
        self._valsize = self._X_val.shape[0]
        self._testsize = self._X_test.shape[0]
        self._construct_provenance()


class FashionMNIST(DirtyLabelDataset, modality=DatasetModality.IMAGE):
    def __init__(
        self,
        trainsize: int = DEFAULT_TRAINSIZE,
        valsize: int = DEFAULT_VALSIZE,
        testsize: int = DEFAULT_TESTSIZE,
        seed: int = DEFAULT_SEED,
        classes: Sequence[int] = DEFAULT_CLASSES,
        **kwargs
    ) -> None:
        self._classes = classes
        super().__init__(trainsize=trainsize, valsize=valsize, testsize=testsize, seed=seed, **kwargs)

    def classes(self) -> Sequence[int]:
        return self._classes

    @classmethod
    def preload(cls) -> None:
        datasets.load_dataset("fashion_mnist", cache_dir=DEFAULT_DATA_DIR)

    def load(self) -> None:
        data = datasets.load_dataset("fashion_mnist", cache_dir=DEFAULT_DATA_DIR)
        assert isinstance(data, datasets.dataset_dict.DatasetDict)

        # Select training and test validation examples based on the provided classes.
        train, test = data["train"], data["test"]
        assert isinstance(train, datasets.arrow_dataset.Dataset) and isinstance(test, datasets.arrow_dataset.Dataset)
        train_val_idx = np.where(np.isin(np.array(train["label"]), self._classes))[0]
        test_idx = np.where(np.isin(np.array(test["label"]), self._classes))[0]

        # Produce random samples of the training, validation and test sets based on the provided set sizes.
        trainsize = self.trainsize if self.trainsize > 0 else None
        valsize = self.valsize if self.valsize > 0 else None
        train_idx, val_idx = train_test_split(
            train_val_idx, train_size=trainsize, test_size=valsize, random_state=self._seed
        )
        if self.testsize > 0:
            random = np.random.RandomState(seed=self._seed)
            test_idx = random.choice(test_idx, size=self.testsize, replace=False)
        train_subset = train.select(train_idx)
        val_subset = train.select(val_idx)
        test_subset = test.select(test_idx)

        # Extract features.
        self._X_train = np.stack(train_subset["image"])
        self._X_val = np.stack(val_subset["image"])
        self._X_test = np.stack(test_subset["image"])
        # TODO: Handle reshaping.

        # Encode labels.
        encoder = LabelEncoder()
        self._y_train = encoder.fit_transform(np.array(train_subset["label"], dtype=int))
        self._y_val = encoder.transform(np.array(val_subset["label"], dtype=int))
        self._y_test = encoder.transform(np.array(test_subset["label"], dtype=int))

        self._loaded = True
        assert self._X_train is not None and self._X_val is not None and self._X_test is not None
        self._trainsize = self._X_train.shape[0]
        self._valsize = self._X_val.shape[0]
        self._testsize = self._X_test.shape[0]
        self._construct_provenance()


class TwentyNewsGroups(DirtyLabelDataset, modality=DatasetModality.TEXT):
    @classmethod
    def preload(cls) -> None:
        categories = ["comp.graphics", "sci.med"]
        fetch_20newsgroups(subset="train", categories=categories, shuffle=True, data_home=DEFAULT_DATA_DIR)
        fetch_20newsgroups(subset="test", categories=categories, shuffle=True, data_home=DEFAULT_DATA_DIR)

    def load(self) -> None:
        categories = ["comp.graphics", "sci.med"]
        train = fetch_20newsgroups(
            subset="train", categories=categories, shuffle=True, random_state=self._seed, data_home=DEFAULT_DATA_DIR
        )
        test = fetch_20newsgroups(
            subset="test", categories=categories, shuffle=True, random_state=self._seed, data_home=DEFAULT_DATA_DIR
        )

        # Load the train and validaiton data by splitting the original training dataset.
        self._X_train, self._y_train = np.array(train.data), np.array(train.target)
        trainsize = self.trainsize if self.trainsize > 0 else None
        valsize = self.valsize if self.valsize > 0 else None
        totsize = self._X_train.shape[0]
        if trainsize is not None and valsize is not None and trainsize + valsize > totsize:
            valsize = totsize - trainsize
        self._X_train, self._X_val, self._y_train, self._y_val = train_test_split(
            self._X_train, self._y_train, train_size=trainsize, test_size=valsize, random_state=self._seed
        )

        # Load the test data.
        self._X_test, self._y_test = np.array(test.data), np.array(test.target)
        if self.testsize > 0:
            self._X_test, self._y_test = self._X_test[: self.testsize], self._y_test[: self.testsize]
        self._loaded = True

        assert self._X_train is not None and self._X_val is not None
        self._trainsize = self._X_train.shape[0]
        self._valsize = self._X_val.shape[0]
        self._testsize = self._X_test.shape[0]
        self._construct_provenance()


class Higgs(DirtyLabelDataset, modality=DatasetModality.TABULAR):

    TRAINSIZES = [1000, 10000, 100000, 1000000]
    TESTSIZES = [500, 5000, 50000, 500000]

    @classmethod
    def preload(cls) -> None:
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz"
        filedir = os.path.join(DEFAULT_DATA_DIR, "higgs")
        os.makedirs(filedir, exist_ok=True)
        filename = os.path.join(filedir, "HIGGS.csv.gz")
        if os.path.exists(filename):
            return
        urllib.request.urlretrieve(url, filename=filename)
        df = pd.read_csv(filename, compression="gzip", header=None)
        df_tr, df_ts = df[:-500000], df[-500000:]

        df_tr.to_csv(os.path.join(filedir, "HIGGS.train.csv.gz"), header=False, index=False, compression="gzip")
        df_ts.to_csv(os.path.join(filedir, "HIGGS.test.csv.gz"), header=False, index=False, compression="gzip")
        for n_tr, n_ts in zip(Higgs.TRAINSIZES, Higgs.TESTSIZES):
            df_tr_s = df_tr.sample(n=n_tr, random_state=7)
            df_ts_s = df_ts.sample(n=n_ts, random_state=7)
            filename_tr = os.path.join(filedir, "HIGGS.train.%d.csv.gz" % n_tr)
            filename_ts = os.path.join(filedir, "HIGGS.test.%d.csv.gz" % n_ts)
            df_tr_s.to_csv(filename_tr, header=False, index=False, compression="gzip")
            df_ts_s.to_csv(filename_ts, header=False, index=False, compression="gzip")

    def load(self) -> None:
        trainsize = self.trainsize if self.trainsize > 0 else None
        valsize = self.valsize if self.valsize > 0 else None
        trainvalsize = trainsize + valsize if trainsize is not None and valsize is not None else None
        testsize = self.testsize if self.testsize > 0 else None
        n_tr: Optional[int] = None
        n_ts: Optional[int] = None
        if trainvalsize is not None:
            sizes = [n for n in Higgs.TRAINSIZES if n >= trainvalsize]
            n_tr = sizes[0] if len(sizes) > 0 else None
        if testsize is not None:
            sizes = [n for n in Higgs.TESTSIZES if n >= testsize]
            n_ts = sizes[0] if len(sizes) > 0 else None

        filedir = os.path.join(DEFAULT_DATA_DIR, "higgs")
        filename_tr = (
            os.path.join(filedir, "HIGGS.train.%d.csv.gz" % n_tr)
            if n_tr is not None
            else os.path.join(filedir, "HIGGS.train.csv.gz")
        )
        filename_ts = (
            os.path.join(filedir, "HIGGS.test.%d.csv.gz" % n_ts)
            if n_ts is not None
            else os.path.join(filedir, "HIGGS.test.csv.gz")
        )
        df_tr = pd.read_csv(filename_tr, compression="gzip", header=None)
        df_ts = pd.read_csv(filename_ts, compression="gzip", header=None)

        y, X = df_tr.iloc[:, 0].to_numpy(dtype=int), df_tr.iloc[:, 1:].to_numpy()
        self._X_train, self._X_val, self._y_train, self._y_val = train_test_split(
            X, y, train_size=trainsize, test_size=valsize, random_state=self._seed
        )

        if testsize is not None:
            df_ts = df_ts.sample(n=self.testsize, random_state=self._seed)
        self._y_test, self._X_test = df_ts.iloc[:, 0].to_numpy(dtype=int), df_ts.iloc[:, 1:].to_numpy()

        assert self._X_train is not None and self._X_val is not None
        self._loaded = True
        self._trainsize = self._X_train.shape[0]
        self._valsize = self._X_val.shape[0]
        self._testsize = self._X_test.shape[0]
        self._construct_provenance()


def preload_datasets(**kwargs) -> None:
    for cls in tqdm(Dataset.datasets.values()):
        tqdm.write("Loading dataset '%s'." % cls._dataset)
        cls.preload()
