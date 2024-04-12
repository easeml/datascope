from __future__ import absolute_import

import collections
import datasets
import functools
import inspect
import math
import numpy as np
import os
import pandas as pd
import pyarrow
import shutil
import tables as tb
import torch

from abc import ABC, abstractmethod
from copy import deepcopy
from datascope.utility import Provenance
from enum import Enum
from folktables import ACSDataSource, ACSIncome
from glob import glob
from hashlib import md5
from joblib import Memory
from joblib.hashing import NumpyHasher
from math import ceil, floor
from numpy.typing import NDArray
from pandas import DataFrame, Series, Index
from PIL import Image
from pyarrow import parquet as pq
from scipy.sparse import issparse, spmatrix
from sklearn.datasets import fetch_openml, fetch_20newsgroups, make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from tqdm.auto import tqdm
from typing import Dict, List, Optional, Sequence, Tuple, Type, Union, Callable, Hashable

from ..utility import download, untar, unzip, unpickle


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
DEFAULT_TESTSIZE = 100
DEFAULT_NUMFEATURES = 10
# DEFAULT_TRAINSIZE = 100
# DEFAULT_VALSIZE = 20
DEFAULT_SEED = 1
DEFAULT_CLASSES = [0, 6]
DEFAULT_DATA_DIR = os.path.join("var", "data")
DEFAULT_CACHE_DIR = os.path.join(DEFAULT_DATA_DIR, "applycache")
DEFAULT_BATCH_SIZE = 1024


memory = Memory(DEFAULT_CACHE_DIR, verbose=0)


def cache(
    func: Optional[Callable] = None,
    memory: Optional[Memory] = None,
    prehash: Optional[List[str]] = None,
    ignore: Optional[List[str]] = None,
) -> Callable:
    # If the function was called as a decorator, we return a partial decorating function.
    if func is None:
        return functools.partial(cache, memory=memory, prehash=prehash, ignore=ignore)

    # By default we instantiate a default memory object.
    if memory is None:
        memory = Memory()

    # We combine the ignore list with the list of arguments to prehash.
    prehash = [] if prehash is None else prehash
    ignore = ([] if ignore is None else ignore) + prehash
    cached_func = memory.cache(func=func, ignore=ignore)

    # Extract arguments and default values.
    arg_sig = inspect.signature(func)
    arg_names = []
    arg_defaults = []
    for param in arg_sig.parameters.values():
        if param.kind is param.POSITIONAL_OR_KEYWORD or param.kind is param.KEYWORD_ONLY:
            arg_names.append(param.name)
            default = param.default if param.default is not param.empty else None
            arg_defaults.append(default)

    def wrapper(*args, **kwargs):
        # Handle arguments that we need to prehash.
        args = list(args)
        targets = {}
        for arg_position, arg_name in enumerate(arg_names):
            if arg_name in prehash:
                if arg_position < len(args):
                    targets[arg_name] = args[arg_position]
                elif arg_name in kwargs:
                    targets[arg_name] = kwargs[arg_name]
                elif arg_defaults[arg_position] is not None:
                    targets[arg_name] = arg_defaults[arg_position]
        for name, value in targets.items():
            prehash_op = getattr(value, "__hash_string__", None)
            if callable(prehash_op):
                key = "_%s_hash" % name
                kwargs[key] = prehash_op()

        return cached_func(*args, **kwargs)

    return wrapper


# @cache(memory=memory, prehash=["dataset", "pipeline"])
def _apply_pipeline(dataset: "Dataset", pipeline: Pipeline, **kwargs) -> "Dataset":
    dataset = deepcopy(dataset)
    pipeline = deepcopy(pipeline)
    dataset._X_train = pipeline.fit_transform(dataset.X_train, dataset.y_train)
    dataset._X_val = pipeline.transform(dataset.X_val)
    dataset._X_test = pipeline.transform(dataset.X_test)
    dataset._fresh = False
    return dataset


def batched_pipeline_transform(
    X: NDArray, pipeline: Pipeline, batch_size: int = DEFAULT_BATCH_SIZE, desc: Optional[str] = None
) -> NDArray:
    idx_max = X.shape[0]
    X_batches: List[NDArray] = []
    pbar = tqdm(total=idx_max, desc=desc)
    for idx_from in range(0, idx_max, batch_size):
        idx_to = min(idx_from + batch_size, idx_max)
        X_batch = pipeline.transform(X[idx_from:idx_to])
        if issparse(X_batch):
            assert isinstance(X_batch, spmatrix)
            X_batch = X_batch.todense()
        X_batches.append(X_batch)
        pbar.update(idx_to - idx_from)
    pbar.close()
    return np.concatenate(X_batches, axis=0)


def save_cached_features(X: NDArray, targetdir: str, name: str) -> str:
    shape, sctype = X.shape[1:], X.dtype.str
    filename = os.path.join(targetdir, "%s.hdf5" % name)
    file = tb.open_file(filename, "w")
    table = file.create_table(
        file.root, "apply_cache", description={"features": tb.Col.from_sctype(sctype, shape=shape)}
    )
    table.append(X)
    file.close()
    tqdm.write("Transformed features saved as %s" % filename)
    return filename


def load_cached_features(idx: NDArray, targetdir: str, name: str) -> NDArray:
    filename = os.path.join(targetdir, "%s.hdf5" % name)
    file = tb.open_file(filename, "r")
    table = file.get_node("/apply_cache")
    result = np.array(table.read_coordinates(idx, field="features"))
    file.close()
    return result


class Dataset(ABC):
    datasets: Dict[str, Type["Dataset"]] = {}
    summaries: Dict[str, str] = {}
    _dataset: Optional[str] = None

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
        self._idx_train: Optional[Index] = None
        self._X_train: Optional[NDArray] = None
        self._y_train: Optional[NDArray] = None
        self._metadata_train: Optional[DataFrame] = None
        self._idx_val: Optional[Index] = None
        self._X_val: Optional[NDArray] = None
        self._y_val: Optional[NDArray] = None
        self._metadata_val: Optional[DataFrame] = None
        self._idx_test: Optional[Index] = None
        self._X_test: Optional[NDArray] = None
        self._y_test: Optional[NDArray] = None
        self._metadata_test: Optional[DataFrame] = None
        self._provenance: Optional[Provenance] = None
        self._units: Optional[NDArray] = None
        self._fresh = True

    def __init_subclass__(
        cls: Type["Dataset"],
        abstract: bool = False,
        id: Optional[str] = None,
        summary: Optional[str] = None,
    ) -> None:
        if abstract:
            return
        cls._dataset = id if id is not None else cls.__name__
        Dataset.datasets[cls._dataset] = cls
        if summary is not None:
            Dataset.summaries[cls._dataset] = summary

    @classmethod
    def preload(cls) -> None:
        pass

    @abstractmethod
    def load(self) -> None:
        pass

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
    def idx_train(self) -> Optional[Index]:
        return self._idx_train

    @property
    def X_train(self) -> NDArray:
        if self._X_train is None:
            raise ValueError("The dataset is not loaded yet.")
        return self._X_train

    @X_train.setter
    def X_train(self, value: NDArray):
        self._X_train = value
        self._fresh = False

    @property
    def y_train(self) -> NDArray:
        if self._y_train is None:
            raise ValueError("The dataset is not loaded yet.")
        return self._y_train

    @y_train.setter
    def y_train(self, value: NDArray):
        self._y_train = value
        self._fresh = False

    @property
    def metadata_train(self) -> Optional[DataFrame]:
        return self._metadata_train

    @property
    def idx_val(self) -> Optional[Index]:
        return self._idx_val

    @property
    def X_val(self) -> NDArray:
        if self._X_val is None:
            raise ValueError("The dataset is not loaded yet.")
        return self._X_val

    @X_val.setter
    def X_val(self, value: NDArray):
        self._X_val = value
        self._fresh = False

    @property
    def y_val(self) -> NDArray:
        if self._y_val is None:
            raise ValueError("The dataset is not loaded yet.")
        return self._y_val

    @y_val.setter
    def y_val(self, value: NDArray):
        self._y_val = value
        self._fresh = False

    @property
    def metadata_val(self) -> Optional[DataFrame]:
        return self._metadata_val

    @property
    def idx_test(self) -> Optional[Index]:
        return self._idx_test

    @property
    def X_test(self) -> NDArray:
        if self._X_test is None:
            raise ValueError("The dataset is not loaded yet.")
        return self._X_test

    @X_test.setter
    def X_test(self, value: NDArray):
        self._X_test = value
        self._fresh = False

    @property
    def y_test(self) -> NDArray:
        if self._y_test is None:
            raise ValueError("The dataset is not loaded yet.")
        return self._y_test

    @y_test.setter
    def y_test(self, value: NDArray):
        self._y_test = value
        self._fresh = False

    @property
    def metadata_test(self) -> Optional[DataFrame]:
        return self._metadata_test

    @property
    def provenance(self) -> Provenance:
        if self._provenance is None:
            raise ValueError("The dataset is not loaded yet.")
        return self._provenance

    @property
    def units(self) -> NDArray:
        if self._units is None:
            raise ValueError("The dataset is not loaded yet.")
        return self._units

    def _construct_provenance(self, groupings: Optional[NDArray] = None) -> None:
        if groupings is None:
            self._units = np.arange(self._trainsize, dtype=int)
            self._provenance = Provenance(units=self._trainsize)
        else:
            self._units = np.unique(groupings)
            self._provenance = Provenance(data=groupings)
            # provenance = np.expand_dims(groupings, axis=(1, 2, 3))
            # self._provenance = np.pad(provenance, pad_width=((0, 0), (0, 0), (0, 0), (0, 1)))

    @classmethod
    def construct_apply_cache(
        cls: Type["Dataset"],
        pipeline: Pipeline,
        cache_dir: str = DEFAULT_CACHE_DIR,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        raise NotImplementedError()

    def query_cached_apply(
        self, pipeline: Pipeline, cache_dir: str = DEFAULT_CACHE_DIR, inplace: bool = False
    ) -> Optional["Dataset"]:
        targetdir = os.path.join(cache_dir, type(self).__name__, str(pipeline))
        if not os.path.isdir(targetdir) or self._idx_train is None or self._idx_val is None or self._idx_test is None:
            return None

        result = self if inplace else deepcopy(self)
        result._X_train = load_cached_features(self._idx_train.to_numpy(), targetdir=targetdir, name="train")
        result._X_val = load_cached_features(self._idx_val.to_numpy(), targetdir=targetdir, name="train")
        result._X_test = load_cached_features(self._idx_test.to_numpy(), targetdir=targetdir, name="test")
        return result

    def apply(self, pipeline: Pipeline, cache_dir: str = DEFAULT_CACHE_DIR, inplace: bool = False) -> "Dataset":
        result = self.query_cached_apply(pipeline=pipeline, cache_dir=cache_dir, inplace=inplace)
        if result is None:
            result = _apply_pipeline(dataset=self, pipeline=pipeline)
            if inplace:
                self._X_train = result._X_train
                self._X_val = result._X_val
                self._X_test = result._X_test
        return result

    def __hash_string__(self) -> str:
        myclass: Type[Dataset] = type(self)
        hash = md5()
        for cls in inspect.getmro(myclass):
            if cls != object:
                hash.update(inspect.getsource(cls).encode("utf-8"))
        if not self._fresh:
            hasher = NumpyHasher()
            hasher.dump(self._X_train)
            hasher.dump(self._y_train)
            hasher.dump(self._X_val)
            hasher.dump(self._y_val)
            hasher.dump(self._X_test)
            hasher.dump(self._y_test)
            hash.update(hasher.stream.getvalue())

        return "%s.%s(trainsize=%d, valsize=%d, testsize=%d, seed=%d, hash=%s)" % (
            type(self).__module__,
            type(self).__name__,
            self.trainsize,
            self.valsize,
            self.testsize,
            self._seed,
            hash.hexdigest(),
        )

    def __repr__(self, N_CHAR_MAX=700):
        return "%s.%s(trainsize=%d, valsize=%d, testsize=%d, seed=%d)" % (
            type(self).__module__,
            type(self).__name__,
            self.trainsize,
            self.valsize,
            self.testsize,
            self._seed,
        )

    def select_train(
        self, index: Union[Sequence[int], Sequence[bool], NDArray[np.int_], NDArray[np.bool_]]
    ) -> "Dataset":
        result = deepcopy(self)
        result.X_train = self.X_train[index]
        result.y_train = self.y_train[index]
        if self._idx_train is not None:
            result._idx_train = self._idx_train[index]
        if self._metadata_train is not None:
            result._metadata_train = self._metadata_train.iloc[np.array(index)]
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


class TabularDatasetMixin:
    @classmethod
    @abstractmethod
    def get_features(cls: Type["TabularDatasetMixin"], element_id: Hashable) -> Series:
        raise NotImplementedError()


class ImageDatasetMixin:
    @classmethod
    @abstractmethod
    def get_features(cls: Type["ImageDatasetMixin"], element_id: Hashable) -> Image:
        raise NotImplementedError()


class TextDatasetMixin:
    @classmethod
    @abstractmethod
    def get_features(cls: Type["TextDatasetMixin"], element_id: Hashable) -> str:
        raise NotImplementedError()


class RandomDataset(Dataset, TabularDatasetMixin, id="random"):
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
        assert self._X_train is not None and self._X_val is not None and self._X_test is not None
        self._trainsize = self._X_train.shape[0]
        self._valsize = self._X_val.shape[0]
        self._testsize = self._X_test.shape[0]
        self._construct_provenance()


class NoisyLabelDataset(Dataset, abstract=True):
    def __init__(
        self,
        trainsize: int = DEFAULT_TRAINSIZE,
        valsize: int = DEFAULT_VALSIZE,
        testsize: int = DEFAULT_TESTSIZE,
        seed: int = DEFAULT_SEED,
        **kwargs
    ) -> None:
        super().__init__(trainsize=trainsize, valsize=valsize, testsize=testsize, seed=seed, **kwargs)
        self._y_train_dirty: Optional[NDArray] = None
        self._units_dirty: Optional[NDArray] = None
        self._groupings: Optional[NDArray] = None

    @property
    def units_dirty(self) -> NDArray:
        if self._X_train is None:
            raise ValueError("The dataset is not loaded yet.")
        if self._units_dirty is None:
            return np.zeros(self.trainsize, dtype=bool)
        else:
            return self._units_dirty

    @property
    def y_train(self) -> NDArray:
        y_train = super().y_train
        if self._y_train_dirty is not None:
            if self.provenance.is_simple:
                y_train = np.where(self.units_dirty, self._y_train_dirty, y_train)
            else:
                idx_dirty = self.provenance.query(self.units_dirty)
                y_train = np.where(idx_dirty, self._y_train_dirty, y_train)
        return y_train

    @y_train.setter
    def y_train(self, value: NDArray):
        self._y_train = value
        self._fresh = False

    def corrupt_labels(self, probabilities: Union[float, Sequence[float]]) -> "NoisyLabelDataset":
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
                idx: NDArray = groupings == i
                n_elements = np.sum(idx)
                idx[idx] = random.choice(a=[False, True], size=(n_elements), p=[1 - p, p])
                dirty_idx[idx] = True
                if idx.sum() > 0:
                    units_dirty[i] = True
                result._y_train_dirty[idx] = 1 - result._y_train[idx]
        result._construct_provenance(groupings=groupings)
        result._groupings = groupings
        result._units_dirty = units_dirty
        result._fresh = False
        return result


class NaturallyNoisyLabelDataset(NoisyLabelDataset, abstract=True):
    def __init__(
        self,
        trainsize: int = DEFAULT_TRAINSIZE,
        valsize: int = DEFAULT_VALSIZE,
        testsize: int = DEFAULT_TESTSIZE,
        seed: int = DEFAULT_SEED,
        **kwargs
    ) -> None:
        super().__init__(trainsize=trainsize, valsize=valsize, testsize=testsize, seed=seed, **kwargs)

    def corrupt_labels(self, probabilities: Union[float, Sequence[float]]) -> "NaturallyNoisyLabelDataset":
        if not self.loaded:
            raise ValueError("The dataset is not loaded yet.")
        result = deepcopy(self)
        assert result._y_train is not None
        assert result._y_train_dirty is not None

        n_examples = result.X_train.shape[0]
        random = np.random.RandomState(seed=self._seed)
        if not isinstance(probabilities, collections.abc.Sequence):
            assert isinstance(probabilities, float)
            dirty_idx = np.not_equal(result._y_train, result._y_train_dirty)
            empirical_probability = np.mean(dirty_idx)

            # If the empirical probability is more than the given probability,
            # we repair some of the labels to make the empirical probability match the given one.
            if empirical_probability > probabilities:
                ratio = probabilities / empirical_probability
                flip_idx = random.choice(a=[False, True], size=(n_examples), p=[1 - ratio, ratio])
                dirty_idx = np.logical_and(dirty_idx, flip_idx)

            units_dirty = dirty_idx
            groupings = None
        else:
            n_groups = len(probabilities)
            n_examples_per_group = ceil(n_examples / float(n_groups))
            groupings = np.tile(np.arange(n_groups), n_examples_per_group)[:n_examples]
            random.shuffle(groupings)
            dirty_idx = np.not_equal(result._y_train, result._y_train_dirty)
            units_dirty = np.zeros(n_groups, dtype=bool)
            for i, p in enumerate(probabilities):
                idx: NDArray = groupings == i
                n_elements = np.sum(idx)
                empirical_probability = np.mean(dirty_idx[idx])
                if empirical_probability > p:
                    ratio = p / empirical_probability
                    flip_idx = random.choice(a=[False, True], size=(n_elements), p=[1 - ratio, ratio])
                    dirty_idx[idx] = np.logical_and(dirty_idx[idx], flip_idx)
                if dirty_idx[idx].sum() > 0:
                    units_dirty[i] = True
            # Repair the labels that should not be dirty anymore.
            result._y_train_dirty[~dirty_idx] = result._y_train[~dirty_idx]

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
        X: NDArray,
        y: NDArray,
        sensitive_feature: int,
        trainsize: int,
        valsize: int,
        testsize: int,
        train_bias: float,
        val_bias: float = 0.0,
        test_bias: float = 0.0,
        bias_method: BiasMethod = DEFAULT_BIAS_METHOD,
        seed: int = DEFAULT_SEED,
    ) -> Tuple[NDArray, NDArray, NDArray]:
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


class BiasedNoisyLabelDataset(NoisyLabelDataset, BiasedMixin, abstract=True):
    def corrupt_labels_with_bias(
        self,
        probabilities: Union[float, Sequence[float]],
        groupbias: float = 0.0,
    ) -> "NoisyLabelDataset":
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


DEFAULT_AUGMENT_FACTOR = 10


class AugmentableMixin:
    @abstractmethod
    def augment(self, factor: int = DEFAULT_AUGMENT_FACTOR, inplace: bool = False) -> Dataset:
        raise NotImplementedError()


UCI_DEFAULT_SENSITIVE_FEATURE = 9
FOLK_UCI_DEFAULT_SENSITIVE_FEATURE = 8


def compute_bias(X: NDArray, y: NDArray, sf: int) -> float:
    n_f0_l0 = np.sum((X[:, sf] == 0) & (y == 0))
    n_f0_l1 = np.sum((X[:, sf] == 0) & (y == 1))
    n_f1_l0 = np.sum((X[:, sf] == 1) & (y == 0))
    n_f1_l1 = np.sum((X[:, sf] == 1) & (y == 1))
    bias = float(n_f0_l0 + n_f1_l1 - n_f0_l1 - n_f1_l0) / (n_f0_l0 + n_f1_l1 + n_f0_l1 + n_f1_l0)
    return bias


class UCI(BiasedNoisyLabelDataset, TabularDatasetMixin, summary="UCI Adult"):
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
        fetch_openml(data_id=1590, as_frame=False, data_home=DEFAULT_DATA_DIR, return_X_y=True)

    def load(self) -> None:
        X: NDArray
        y: NDArray
        X, y = fetch_openml(data_id=1590, as_frame=False, data_home=DEFAULT_DATA_DIR, return_X_y=True)
        X = np.nan_to_num(X)  # TODO: Maybe leave nan values.
        y = np.array(y == ">50K", dtype=int)

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
        totsize = X.shape[0]
        idx_train, idx_test_val = train_test_split(
            np.arange(totsize), train_size=trainsize, test_size=valtestsize, random_state=self._seed
        )
        idx_val, idx_test = train_test_split(
            idx_test_val, train_size=valsize, test_size=testsize, random_state=self._seed
        )
        self._X_train, self._y_train = X[idx_train], y[idx_train]
        self._X_val, self._y_val = X[idx_val], y[idx_val]
        self._X_test, self._y_test = X[idx_test], y[idx_test]
        self._idx_train, self._idx_val, self._idx_test = Index(idx_train), Index(idx_val), Index(idx_test)

        self._loaded = True
        assert self._X_train is not None and self._X_val is not None and self._X_test is not None
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
        X, y = fetch_openml(data_id=1590, as_frame=False, data_home=DEFAULT_DATA_DIR, return_X_y=True)
        X = np.nan_to_num(X)  # TODO: Maybe leave nan values.
        y = np.array(y == ">50K", dtype=int)

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
        self._idx_train, self._idx_val, self._idx_test = Index(idx_train), Index(idx_val), Index(idx_test)
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


class FolkUCI(BiasedNoisyLabelDataset, AugmentableMixin, TabularDatasetMixin, summary="Folktables Adult"):
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
        X: NDArray
        y: NDArray
        X, y, _ = ACSIncome.df_to_numpy(acs_data)
        del acs_data

        trainsize = self.trainsize if self.trainsize > 0 else None
        valsize = self.valsize if self.valsize > 0 else None
        testsize = self.testsize if self.testsize > 0 else None
        valtestsize = valsize + testsize if valsize is not None and testsize is not None else None
        totsize = X.shape[0]
        idx_train, idx_test_val = train_test_split(
            np.arange(totsize), train_size=trainsize, test_size=valtestsize, random_state=self._seed
        )
        idx_val, idx_test = train_test_split(
            idx_test_val, train_size=valsize, test_size=testsize, random_state=self._seed
        )
        self._X_train, self._y_train = X[idx_train], y[idx_train]
        self._X_val, self._y_val = X[idx_val], y[idx_val]
        self._X_test, self._y_test = X[idx_test], y[idx_test]
        self._idx_train, self._idx_val, self._idx_test = Index(idx_train), Index(idx_val), Index(idx_test)
        del X, y

        self._loaded = True
        assert self._X_train is not None and self._X_val is not None and self._X_test is not None
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
        self._idx_train, self._idx_val, self._idx_test = Index(idx_train), Index(idx_val), Index(idx_test)
        assert self._X_train is not None and self._X_val is not None and self._X_test is not None
        self._trainsize = self._X_train.shape[0]
        self._valsize = self._X_val.shape[0]
        self._testsize = self._X_test.shape[0]
        self._construct_provenance()

    def augment(self, factor: int = DEFAULT_AUGMENT_FACTOR, inplace: bool = False) -> Dataset:
        dataset = self if inplace else deepcopy(self)
        if dataset._X_train is None or dataset._y_train is None or dataset._provenance is None:
            raise ValueError("Cannot augment a dataset that is not loaded yet.")
        # trainsize = dataset._X_train.shape[0]
        dataset._X_train = np.repeat(dataset._X_train, repeats=factor, axis=0)
        dataset._y_train = np.repeat(dataset._y_train, repeats=factor, axis=0)
        if dataset._y_train_dirty is not None:
            dataset._y_train_dirty = np.repeat(dataset._y_train_dirty, repeats=factor, axis=0)
        dataset._trainsize = dataset._X_train.shape[0]
        random = np.random.RandomState(seed=self._seed)
        dataset._X_train[:, 0] += random.normal(0, 5, dataset._X_train.shape[0])

        # if checknan(dataset._provenance):
        #     dataset._provenance = np.arange(trainsize)
        # dataset._provenance = reshape(np.repeat(dataset._provenance, repeats=factor, axis=0))
        dataset._provenance = dataset._provenance.fork(size=factor)

        return dataset


class FashionMNIST(NoisyLabelDataset, ImageDatasetMixin):
    CLASSES = ["t-shirt/top", "trouser", "pullover", "dress", "coat", "sandal", "shirt", "sneaker", "bag", "ankle boot"]

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
        idx_train_val = np.where(np.isin(np.array(train["label"]), self._classes))[0]
        idx_test = np.where(np.isin(np.array(test["label"]), self._classes))[0]

        # Produce random samples of the training, validation and test sets based on the provided set sizes.
        trainsize = self.trainsize if self.trainsize > 0 else None
        valsize = self.valsize if self.valsize > 0 else None
        idx_train, idx_val = train_test_split(
            idx_train_val, train_size=trainsize, test_size=valsize, random_state=self._seed
        )
        if self.testsize > 0:
            random = np.random.RandomState(seed=self._seed)
            idx_test = random.choice(idx_test, size=self.testsize, replace=False)
        train_subset = train.select(idx_train)
        val_subset = train.select(idx_val)
        test_subset = test.select(idx_test)
        self._idx_train, self._idx_val, self._idx_test = Index(idx_train), Index(idx_val), Index(idx_test)

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

    @classmethod
    def construct_apply_cache(
        cls: Type["Dataset"],
        pipeline: Pipeline,
        cache_dir: str = DEFAULT_CACHE_DIR,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        # Determine and create the target directory. If it already exists, remove it first.
        targetdir = os.path.join(cache_dir, cls.__name__, str(pipeline))
        if os.path.isdir(targetdir):
            shutil.rmtree(targetdir)
        os.makedirs(targetdir)

        data = datasets.load_dataset("fashion_mnist", cache_dir=DEFAULT_DATA_DIR)
        train, test = data["train"], data["test"]
        assert isinstance(train, datasets.arrow_dataset.Dataset) and isinstance(test, datasets.arrow_dataset.Dataset)
        X_train, y_train = np.stack(train["image"]), np.array(train["label"], dtype=int)
        X_test = np.stack(test["image"])
        pipeline = deepcopy(pipeline)
        pipeline.fit(X_train, y_train)
        X_train_transformed = batched_pipeline_transform(
            X_train, pipeline, batch_size=batch_size, desc="Train data examples"
        )
        save_cached_features(X_train_transformed, targetdir=targetdir, name="train")
        X_test_transformed = batched_pipeline_transform(
            X_test, pipeline, batch_size=batch_size, desc="Test data examples"
        )
        save_cached_features(X_test_transformed, targetdir=targetdir, name="test")


class TwentyNewsGroups(NoisyLabelDataset, TextDatasetMixin, summary="20NewsGroups"):
    @classmethod
    def preload(cls) -> None:
        categories = ["comp.graphics", "sci.med"]
        fetch_20newsgroups(subset="train", categories=categories, shuffle=False, data_home=DEFAULT_DATA_DIR)
        fetch_20newsgroups(subset="test", categories=categories, shuffle=False, data_home=DEFAULT_DATA_DIR)

    def load(self) -> None:
        categories = ["comp.graphics", "sci.med"]
        train = fetch_20newsgroups(subset="train", categories=categories, shuffle=False, data_home=DEFAULT_DATA_DIR)
        test = fetch_20newsgroups(subset="test", categories=categories, shuffle=False, data_home=DEFAULT_DATA_DIR)

        # Load the train and validaiton data by splitting the original training dataset. Load the test data.
        X, y = np.array(train.data), np.array(train.target)
        self._X_test, self._y_test = np.array(test.data), np.array(test.target)
        trainsize = self.trainsize if self.trainsize > 0 else None
        valsize = self.valsize if self.valsize > 0 else None
        totsize = X.shape[0]
        if trainsize is not None and valsize is not None and trainsize + valsize > totsize:
            valsize = totsize - trainsize
        idx_train, idx_val = train_test_split(
            np.arange(totsize), train_size=trainsize, test_size=valsize, random_state=self._seed
        )
        idx_test = np.arange(self._X_test.shape[0])
        if self._testsize > 0:
            random = np.random.RandomState(seed=self._seed)
            idx_test = random.choice(idx_test, size=self._testsize, replace=False)
            self._X_test, self._y_test = self._X_test[idx_test], self._y_test[idx_test]

        self._X_train, self._y_train = X[idx_train], y[idx_train]
        self._X_val, self._y_val = X[idx_val], y[idx_val]
        self._idx_train, self._idx_val, self._idx_test = Index(idx_train), Index(idx_val), Index(idx_test)

        self._loaded = True

        assert self._X_train is not None and self._X_val is not None and self._X_test is not None
        self._trainsize = self._X_train.shape[0]
        self._valsize = self._X_val.shape[0]
        self._testsize = self._X_test.shape[0]
        self._construct_provenance()

    @classmethod
    def construct_apply_cache(
        cls: Type["Dataset"],
        pipeline: Pipeline,
        cache_dir: str = DEFAULT_CACHE_DIR,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        # Determine and create the target directory. If it already exists, remove it first.
        targetdir = os.path.join(cache_dir, cls.__name__, str(pipeline))
        if os.path.isdir(targetdir):
            shutil.rmtree(targetdir)
        os.makedirs(targetdir)

        categories = ["comp.graphics", "sci.med"]
        train = fetch_20newsgroups(subset="train", categories=categories, shuffle=False, data_home=DEFAULT_DATA_DIR)
        test = fetch_20newsgroups(subset="test", categories=categories, shuffle=False, data_home=DEFAULT_DATA_DIR)
        X_train, y_train = np.array(train.data), np.array(train.target)
        X_test = np.array(test.data)

        pipeline = deepcopy(pipeline)
        pipeline.fit(X_train, y_train)
        X_train_transformed = batched_pipeline_transform(
            X_train, pipeline, batch_size=batch_size, desc="Train data examples"
        )
        save_cached_features(X_train_transformed, targetdir=targetdir, name="train")
        X_test_transformed = batched_pipeline_transform(
            X_test, pipeline, batch_size=batch_size, desc="Test data examples"
        )
        save_cached_features(X_test_transformed, targetdir=targetdir, name="test")


class Higgs(NoisyLabelDataset, TabularDatasetMixin):
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
        download(url=url, filename=filename)
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
        totsize = X.shape[0]
        idx_train, idx_val = train_test_split(
            np.arange(totsize), train_size=trainsize, test_size=valsize, random_state=self._seed
        )
        self._X_train, self._y_train = X[idx_train], y[idx_train]
        self._X_val, self._y_val = X[idx_val], y[idx_val]
        self._idx_train, self._idx_val = Index(idx_train), Index(idx_val)

        y_test, X_test = df_ts.iloc[:, 0].to_numpy(dtype=int), df_ts.iloc[:, 1:].to_numpy()
        random = np.random.RandomState(seed=self._seed)
        idx_test = np.arange(X_test.shape[0])
        if self.testsize > 0:
            idx_test = random.permutation(self.testsize)
        self._X_test = X_test[idx_test, :]
        self._y_test = y_test[idx_test]
        self._idx_test = Index(idx_test)

        assert self._X_train is not None and self._X_val is not None and self._X_test is not None
        self._loaded = True
        self._trainsize = self._X_train.shape[0]
        self._valsize = self._X_val.shape[0]
        self._testsize = self._X_test.shape[0]
        self._construct_provenance()


class DataPerfVision(NaturallyNoisyLabelDataset, TabularDatasetMixin, summary="DataPerf Vision"):
    DATA_DIR = os.path.join(DEFAULT_DATA_DIR, "dataperf-vision")

    @classmethod
    def preload(cls: Type["DataPerfVision"]) -> None:
        # Download and extract the source zip file.
        url = "https://drive.google.com/uc?export=download&id=1_-WZCGd31XTENCtVjP4GqDLYShP3NzwK&confirm=t"
        os.makedirs(cls.DATA_DIR, exist_ok=True)
        filename = os.path.join(cls.DATA_DIR, "dataperf-vision.zip")
        if not os.path.exists(filename):
            download(url=url, filename=filename)
            unzip(source=filename, path=cls.DATA_DIR)

        # If all numpy files already exist, we can return.
        numpy_files = {
            "X_train": os.path.join(cls.DATA_DIR, "X_train.npy"),
            "X_val": os.path.join(cls.DATA_DIR, "X_val.npy"),
            "X_test": os.path.join(cls.DATA_DIR, "X_test.npy"),
            "y_train": os.path.join(cls.DATA_DIR, "y_train.npy"),
            "y_train_dirty": os.path.join(cls.DATA_DIR, "y_train_dirty.npy"),
            "y_val": os.path.join(cls.DATA_DIR, "y_val.npy"),
            "y_test": os.path.join(cls.DATA_DIR, "y_test.npy"),
        }
        if all(os.path.exists(p) for p in numpy_files.values()):
            return

        # Collect dataset ID's and find all relevan file paths.
        dataset_ids = [
            os.path.basename(x.split("_train")[0])
            for x in glob(os.path.join(cls.DATA_DIR, "embeddings", "*_train*"))
            if "flipped" not in x
        ]
        X_train_paths = [os.path.join(cls.DATA_DIR, "embeddings", "%s_train_0.3_300.parquet" % x) for x in dataset_ids]
        X_val_paths = [os.path.join(cls.DATA_DIR, "embeddings", "%s_val_100.parquet" % x) for x in dataset_ids]
        X_test_paths = [os.path.join(cls.DATA_DIR, "embeddings", "%s_test_500.parquet" % x) for x in dataset_ids]
        y_train_paths = [os.path.join(cls.DATA_DIR, "data", "dataset_%s_train.csv" % x) for x in dataset_ids]
        y_val_paths = [os.path.join(cls.DATA_DIR, "data", "dataset_%s_val.csv" % x) for x in dataset_ids]
        y_test_paths = [os.path.join(cls.DATA_DIR, "data", "dataset_%s_test.csv" % x) for x in dataset_ids]

        # Load feature tables and extract data example indices (keys).
        X_train_table = pyarrow.concat_tables([pq.read_table(path) for path in X_train_paths])
        X_val_table = pyarrow.concat_tables([pq.read_table(path) for path in X_val_paths])
        X_test_table = pyarrow.concat_tables([pq.read_table(path) for path in X_test_paths])
        idx_train = set(np.vstack(X_train_table.column("filename").to_numpy()).flatten().tolist())
        idx_val = set(np.vstack(X_val_table.column("filename").to_numpy()).flatten().tolist())
        idx_test = set(np.vstack(X_test_table.column("filename").to_numpy()).flatten().tolist())

        # Ensure no index appears in more than one set.
        idx_train = idx_train - idx_val
        idx_test = idx_test - idx_val
        idx_test = idx_test - idx_train

        # Transfer 601 examples from the test set into the validation set so it can have 1000 examples.
        idx_transfer = sorted(idx_test)
        random = np.random.RandomState(seed=1)
        random.shuffle(idx_transfer)
        idx_test = idx_test.difference(idx_transfer[:601])
        idx_val = idx_val.union(idx_transfer[:601])

        # Convert sets of indices to row selectors.
        X_table = pyarrow.concat_tables([X_train_table, X_val_table, X_test_table])
        X_idx = X_table.column("filename").to_pandas()
        X_idx_dups = X_idx.duplicated(keep="first")
        X_train_selector = X_idx.isin(idx_train) & ~X_idx_dups
        X_val_selector = X_idx.isin(idx_val) & ~X_idx_dups
        X_test_selector = X_idx.isin(idx_test) & ~X_idx_dups

        # Select the train, validation and test features from feature tables.
        X_train = np.vstack(
            X_table.filter(mask=X_train_selector.to_list())
            .select(["filename", "encoding"])
            .sort_by("filename")
            .column("encoding")
            .to_numpy()
        )
        X_val = np.vstack(
            X_table.filter(mask=X_val_selector.to_list())
            .select(["filename", "encoding"])
            .sort_by("filename")
            .column("encoding")
            .to_numpy()
        )
        X_test = np.vstack(
            X_table.filter(mask=X_test_selector.to_list())
            .select(["filename", "encoding"])
            .sort_by("filename")
            .column("encoding")
            .to_numpy()
        )

        # Load the clean and dirty labels from CSV files.
        y_clean = pd.concat(
            [pd.read_csv(path, index_col=0)["hv_label"] for path in y_train_paths + y_val_paths + y_test_paths]
        )
        y_dirty = pd.concat(
            [pd.read_csv(path, index_col=0)["mg_label"] for path in y_train_paths + y_val_paths + y_test_paths]
        )
        y_table_stacked = pd.concat([y_clean, y_dirty], axis=1)
        y_table = y_table_stacked[~y_table_stacked.index.duplicated(keep="first")]

        # Select the labels given the training, validation and test set indices.
        y_train_table = y_table.loc[list(idx_train)].sort_index()
        y_val_table = y_table.loc[list(idx_val)].sort_index()
        y_test_table = y_table.loc[list(idx_test)].sort_index()

        y_train_clean = y_train_table["hv_label"].to_numpy()
        y_train_dirty = y_train_table["mg_label"].to_numpy()

        y_val_clean = y_val_table["hv_label"].to_numpy()
        y_test_clean = y_test_table["hv_label"].to_numpy()

        # Shuffle the datasets.
        permutation_train = random.permutation(X_train.shape[0])
        permutation_val = random.permutation(X_val.shape[0])
        permutation_test = random.permutation(X_test.shape[0])
        X_train = X_train[permutation_train, :]
        X_val = X_val[permutation_val, :]
        X_test = X_test[permutation_test, :]
        y_train_clean = y_train_clean[permutation_train]
        y_train_dirty = y_train_dirty[permutation_train]
        y_val_clean = y_val_clean[permutation_val]
        y_test_clean = y_test_clean[permutation_test]

        # Save all the feature and label numpy arrays.
        np.save(numpy_files["X_train"], X_train)
        np.save(numpy_files["X_val"], X_val)
        np.save(numpy_files["X_test"], X_test)
        np.save(numpy_files["y_train"], y_train_clean)
        np.save(numpy_files["y_train_dirty"], y_train_dirty)
        np.save(numpy_files["y_val"], y_val_clean)
        np.save(numpy_files["y_test"], y_test_clean)

    def load(self) -> None:
        self._X_train = np.load(os.path.join(self.DATA_DIR, "X_train.npy"))
        self._X_val = np.load(os.path.join(self.DATA_DIR, "X_val.npy"))
        self._X_test = np.load(os.path.join(self.DATA_DIR, "X_test.npy"))
        self._y_train = np.load(os.path.join(self.DATA_DIR, "y_train.npy"))
        self._y_train_dirty = np.load(os.path.join(self.DATA_DIR, "y_train_dirty.npy"))
        self._y_val = np.load(os.path.join(self.DATA_DIR, "y_val.npy"))
        self._y_test = np.load(os.path.join(self.DATA_DIR, "y_test.npy"))
        self._idx_train = Index(np.arange(self.trainsize))
        self._idx_val = Index(np.arange(self.valsize))
        self._idx_test = Index(np.arange(self.testsize))

        assert self._X_train is not None
        assert self._X_val is not None
        assert self._X_test is not None
        assert self._y_train is not None
        assert self._y_train_dirty is not None
        assert self._y_val is not None
        assert self._y_test is not None

        random = np.random.RandomState(seed=self._seed)

        if self.trainsize > 0:
            idx = random.permutation(self.trainsize)
            self._X_train = self._X_train[idx, :]
            self._y_train = self._y_train[idx]
            self._y_train_dirty = self._y_train_dirty[idx]
            self._idx_train = Index(idx)
        if self.valsize > 0:
            idx = random.permutation(self.valsize)
            self._X_val = self._X_val[idx, :]
            self._y_val = self._y_val[idx]
            self._idx_val = Index(idx)
        if self.testsize > 0:
            idx = random.permutation(self.testsize)
            self._X_test = self._X_test[idx, :]
            self._y_test = self._y_test[idx]
            self._idx_test = Index(idx)

        self._loaded = True
        assert self._X_train is not None and self._X_val is not None
        self._trainsize = self._X_train.shape[0]
        self._valsize = self._X_val.shape[0]
        self._testsize = self._X_test.shape[0]
        self._construct_provenance()


class CifarN(NaturallyNoisyLabelDataset, ImageDatasetMixin):
    DATA_DIR = os.path.join(DEFAULT_DATA_DIR, "cifar-n")
    CIFAR_10_DATA_DIR = os.path.join(DATA_DIR, "cifar-10-batches-py")
    CIFAR_10_TRAIN_FILES = ["data_batch_%d" % i for i in [1, 2, 3, 4, 5]]
    CIFAR_10_TEST_FILE = "test_batch"
    CIFAR_N_DATA_DIR = os.path.join(DATA_DIR, "CIFAR-N")
    CIFAR_10N_FILE = "CIFAR-10_human.pt"
    CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    @classmethod
    def preload(cls: Type["CifarN"]) -> None:
        # If all files exist, we can skip.
        filelist = [os.path.join(cls.CIFAR_10_DATA_DIR, batch) for batch in cls.CIFAR_10_TRAIN_FILES]
        filelist += [os.path.join(cls.CIFAR_10_DATA_DIR, cls.CIFAR_10_TEST_FILE)]
        filelist += [os.path.join(cls.CIFAR_N_DATA_DIR, cls.CIFAR_10N_FILE)]
        if all(os.path.exists(p) for p in filelist):
            return

        # Download the CIFAR-10 dataset.
        cifar_url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
        os.makedirs(cls.DATA_DIR, exist_ok=True)
        filename = os.path.join(cls.DATA_DIR, "cifar-10-python.tar.gz")
        if not os.path.exists(filename):
            download(url=cifar_url, filename=filename)
            untar(source=filename, path=cls.DATA_DIR, mode="r:gz")

        # Download the CIFAR-N noisy labels.
        labels_url = "http://www.yliuu.com/web-cifarN/files/CIFAR-N.zip"
        filename = os.path.join(cls.DATA_DIR, "CIFAR-N.zip")
        if not os.path.exists(filename):
            download(url=labels_url, filename=filename)
            unzip(source=filename, path=cls.DATA_DIR)

    def load(self) -> None:
        # Load the CIFAR-10 dataset. We load the minimal number of batches needed.
        num_batches = math.ceil((self.trainsize + self.valsize) / 10000)
        if num_batches == 0:
            num_batches = len(self.CIFAR_10_TRAIN_FILES)
        train_filepaths = [
            os.path.join(self.CIFAR_10_DATA_DIR, batch) for batch in self.CIFAR_10_TRAIN_FILES[:num_batches]
        ]
        train_batches = [unpickle(filepath) for filepath in train_filepaths]
        test_batch = unpickle(os.path.join(self.CIFAR_10_DATA_DIR, self.CIFAR_10_TEST_FILE))
        X_train = np.concatenate(
            [batch["data"].reshape((-1, 3, 32, 32)).transpose((0, 2, 3, 1)) for batch in train_batches]
        )
        y_train = np.concatenate([batch["labels"] for batch in train_batches])
        X_test = test_batch["data"].reshape((-1, 3, 32, 32)).transpose((0, 2, 3, 1))
        y_test = np.array(test_batch["labels"])

        # Load the CIFAR-N noisy labels.
        noisylabels = torch.load(os.path.join(self.CIFAR_N_DATA_DIR, self.CIFAR_10N_FILE))
        y_train_dirty = noisylabels["worse_label"]

        # Perform the train-test split.
        trainsize = self.trainsize if self.trainsize > 0 else None
        valsize = self.valsize if self.valsize > 0 else None
        totsize = X_train.shape[0]
        idx_train, idx_val = train_test_split(
            np.arange(totsize),
            train_size=trainsize,
            test_size=valsize,
            random_state=self._seed,
            stratify=y_train,
        )
        self._X_train = X_train[idx_train]
        self._X_val = X_train[idx_val]
        self._y_train = y_train[idx_train]
        self._y_train_dirty = y_train_dirty[idx_train]
        self._y_val = y_train[idx_val]
        self._idx_train, self._idx_val = Index(idx_train), Index(idx_val)

        # Select the test dataset.
        random = np.random.RandomState(seed=self._seed)
        idx_test = np.arange(X_test.shape[0])
        if self.testsize > 0:
            idx_test = random.permutation(self.testsize)
        self._X_test = X_test[idx_test, :]
        self._y_test = y_test[idx_test]
        self._idx_test = Index(idx_test)

        assert self._X_train is not None and self._X_val is not None and self._X_test is not None
        self._trainsize = self._X_train.shape[0]
        self._valsize = self._X_val.shape[0]
        self._testsize = self._X_test.shape[0]
        self._loaded = True
        self._construct_provenance()

    @classmethod
    def construct_apply_cache(
        cls: Type["CifarN"],
        pipeline: Pipeline,
        cache_dir: str = DEFAULT_CACHE_DIR,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        # Determine and create the target directory. If it already exists, remove it first.
        targetdir = os.path.join(cache_dir, cls.__name__, str(pipeline))
        if os.path.isdir(targetdir):
            shutil.rmtree(targetdir)
        os.makedirs(targetdir)

        train_filepaths = [os.path.join(cls.CIFAR_10_DATA_DIR, batch) for batch in cls.CIFAR_10_TRAIN_FILES]
        train_batches = [unpickle(filepath) for filepath in train_filepaths]
        test_batch = unpickle(os.path.join(cls.CIFAR_10_DATA_DIR, cls.CIFAR_10_TEST_FILE))
        X_train = np.concatenate(
            [batch["data"].reshape((-1, 3, 32, 32)).transpose((0, 2, 3, 1)) for batch in train_batches]
        )
        y_train = np.concatenate([batch["labels"] for batch in train_batches])
        X_test = test_batch["data"].reshape((-1, 3, 32, 32)).transpose((0, 2, 3, 1))

        pipeline = deepcopy(pipeline)
        pipeline.fit(X_train, y_train)
        X_train_transformed = batched_pipeline_transform(
            X_train, pipeline, batch_size=batch_size, desc="Train data examples"
        )
        save_cached_features(X_train_transformed, targetdir=targetdir, name="train")
        X_test_transformed = batched_pipeline_transform(
            X_test, pipeline, batch_size=batch_size, desc="Test data examples"
        )
        save_cached_features(X_test_transformed, targetdir=targetdir, name="test")


def preload_datasets(**kwargs) -> None:
    for cls in tqdm(Dataset.datasets.values()):
        tqdm.write("Loading dataset '%s'." % cls._dataset)
        cls.preload()
