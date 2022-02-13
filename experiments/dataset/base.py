from __future__ import absolute_import

import datasets
import numpy as np

from abc import ABC, abstractmethod
from enum import Enum
from numpy import ndarray
from sklearn.datasets import fetch_openml, fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from typing import Dict, Optional, Sequence, Type


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
    def X_train(self) -> Optional[ndarray]:
        return self._X_train

    @property
    def y_train(self) -> Optional[ndarray]:
        return self._y_train

    @property
    def X_val(self) -> Optional[ndarray]:
        return self._X_val

    @property
    def y_val(self) -> Optional[ndarray]:
        return self._y_val


class UCI(Dataset, modality=DatasetModality.TABULAR):
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
        self._trainsize = self._X_train.shape[0]
        self._valsize = self._X_val.shape[0]
