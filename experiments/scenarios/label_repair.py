import numpy as np
import pandas as pd

from copy import deepcopy
from datascope.importance.common import SklearnModelUtility, binarize, get_indices
from datascope.importance.shapley import ShapleyImportance, ImportanceMethod
from datetime import timedelta
from enum import Enum
from numpy import ndarray
from pandas import DataFrame
from sklearn.metrics import accuracy_score
from time import process_time_ns
from typing import Any, Iterable, Optional, Dict

from .base import Scenario, attribute, result
from ..dataset import Dataset, DEFAULT_TRAINSIZE, DEFAULT_VALSIZE
from ..pipelines import Pipeline, get_model, ModelType


class RepairMethod(str, Enum):
    KNN_Single = "shapley-knn-single"
    KNN_Interactive = "shapley-knn-interactive"
    TMC_1 = "shapley-tmc-001"
    TMC_5 = "shapley-tmc-005"
    TMC_10 = "shapley-tmc-010"
    TMC_50 = "shapley-tmc-050"
    TMC_100 = "shapley-tmc-100"
    TMC_500 = "shapley-tmc-500"
    RANDOM = "random"


IMPORTANCE_METHODS = {
    RepairMethod.KNN_Single: ImportanceMethod.NEIGHBOR,
    RepairMethod.KNN_Interactive: ImportanceMethod.NEIGHBOR,
    RepairMethod.TMC_1: ImportanceMethod.MONTECARLO,
    RepairMethod.TMC_5: ImportanceMethod.MONTECARLO,
    RepairMethod.TMC_10: ImportanceMethod.MONTECARLO,
    RepairMethod.TMC_50: ImportanceMethod.MONTECARLO,
    RepairMethod.TMC_100: ImportanceMethod.MONTECARLO,
    RepairMethod.TMC_500: ImportanceMethod.MONTECARLO,
}


MC_ITERATIONS = {
    RepairMethod.KNN_Single: 0,
    RepairMethod.KNN_Interactive: 0,
    RepairMethod.TMC_1: 1,
    RepairMethod.TMC_5: 5,
    RepairMethod.TMC_10: 10,
    RepairMethod.TMC_50: 50,
    RepairMethod.TMC_100: 100,
    RepairMethod.TMC_500: 500,
}

KEYWORD_REPLACEMENTS = {
    "random": "Random",
    "shapley-tmc": "Shapley TMC",
    "shapley-knn-single": "Shapley KNN Single",
    "shapley-knn-interactive": "Shapley KNN Interactive",
    "shapley-tmc-001": "Shapley TMC x1",
    "shapley-tmc-005": "Shapley TMC x5",
    "shapley-tmc-010": "Shapley TMC x10",
    "shapley-tmc-050": "Shapley TMC x50",
    "shapley-tmc-100": "Shapley TMC x100",
    "shapley-tmc-500": "Shapley TMC x500",
}


DEFAULT_DIRTY_RATIO = 0.5
DEFAULT_SEED = 1
DEFAULT_CHECKPOINTS = 100


class LabelRepairScenario(Scenario, id="label-repair"):
    def __init__(
        self,
        dataset: str,
        pipeline: str,
        method: RepairMethod,
        iteration: int,
        dirty_ratio: float = DEFAULT_DIRTY_RATIO,
        seed: int = DEFAULT_SEED,
        trainsize: int = DEFAULT_TRAINSIZE,
        valsize: int = DEFAULT_VALSIZE,
        checkpoints: int = DEFAULT_CHECKPOINTS,
        evolution: Optional[pd.DataFrame] = None,
        importance_compute_time: Optional[float] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self._dataset = dataset
        self._pipeline = pipeline
        self._method = method
        self._iteration = iteration
        self._dirty_ratio = dirty_ratio
        self._seed = seed
        self._trainsize = trainsize
        self._valsize = valsize
        self._checkpoints = checkpoints
        self._evolution = pd.DataFrame() if evolution is None else evolution
        self._importance_compute_time: Optional[float] = importance_compute_time

    @attribute(domain=Dataset.datasets.keys())
    def dataset(self) -> str:
        """Dataset to use for training and validation."""
        return self._dataset

    @attribute(domain=Pipeline.pipelines.keys())
    def pipeline(self) -> str:
        """Pipeline to use for feature extraction."""
        return self._pipeline

    @attribute
    def method(self) -> RepairMethod:
        """Method used to perform data repairs."""
        return self._method

    @attribute(domain=range(10))
    def iteration(self) -> int:
        """The ordinal number of the experiment repetition. Also serves as the random seed."""
        return self._iteration

    @attribute
    def trainsize(self) -> int:
        """The size of the training dataset to use. The value 0 means maximal value."""
        return self._trainsize

    @attribute
    def valsize(self) -> int:
        """The size of the validation dataset to use. The value 0 means maximal value."""
        return self._valsize

    @attribute
    def checkpoints(self) -> int:
        """
        The number of checkpoints to record in the workflow.
        The value 0 means that a checkpoint will be recorded for each training data example.
        """
        return self._checkpoints

    @result
    def evolution(self) -> DataFrame:
        """The evolution of the experimental parameters."""
        return self._evolution

    @result
    def importance_compute_time(self) -> Optional[float]:
        """The time it takes to compute importance."""
        return self._importance_compute_time

    def _run(self, progress_bar: bool = True, **kwargs: Any) -> None:

        # Load dataset.
        dataset = Dataset.datasets[self.dataset](trainsize=self.trainsize, valsize=self.valsize)
        dataset.load()
        self.logger.debug(
            "Dataset '%s' loaded (trainsize=%d, valsize=%d).", self.dataset, dataset.trainsize, dataset.valsize
        )

        # Create the dirty dataset and apply the data corruption.
        dataset_dirty = deepcopy(dataset)
        random = np.random.RandomState(seed=self._seed + self._iteration)
        dirty_probs = [1 - self._dirty_ratio, self._dirty_ratio]
        dirty_idx = random.choice(a=[False, True], size=(dataset_dirty.trainsize), p=dirty_probs)
        assert dataset_dirty.y_train is not None
        dataset_dirty.y_train[dirty_idx] = 1 - dataset_dirty.y_train[dirty_idx]

        # Load the pipeline and process the data.
        pipeline_class = Pipeline.pipelines[self.pipeline]
        pipeline = pipeline_class.construct(dataset)
        assert dataset.X_train is not None
        X_train: ndarray = pipeline.fit_transform(
            dataset.X_train, dataset.y_train
        )  # TODO: Fit the pipeline with dirty data.
        X_train_dirty: ndarray = pipeline.transform(dataset_dirty.X_train)
        y_train, y_val, y_train_dirty = dataset.y_train, dataset.y_val, dataset_dirty.y_train
        X_val: ndarray = pipeline.transform(dataset.X_val)
        assert y_train is not None
        self.logger.debug("Shape of X_train before feature extraction: %s", str(dataset.X_train.shape))
        self.logger.debug("Shape of X_train after feature extraction: %s", str(X_train.shape))

        # Reshape datasets if needed.
        if X_train.ndim > 2:
            X_train = X_train.reshape(X_train.shape[0], -1)
            X_train_dirty = X_train_dirty.reshape(X_train_dirty.shape[0], -1)
            X_val = X_val.reshape(X_val.shape[0], -1)
            self.logger.debug("Need to reshape. New shape: %s", str(X_train.shape))

        # Construct binarized provenance matrix.
        provenance = np.expand_dims(np.arange(dataset.trainsize, dtype=int), axis=(1, 2, 3))
        provenance = np.pad(provenance, pad_width=((0, 0), (0, 0), (0, 0), (0, 1)))
        provenance = binarize(provenance)

        # Initialize the model and utility.
        model = get_model(ModelType.LogisticRegression)
        utility = SklearnModelUtility(model, accuracy_score)

        # Compute importance scores and time it.
        importance_time_start = process_time_ns()
        importance: Optional[ShapleyImportance] = None
        importances: Optional[Iterable[float]] = None
        if self.method == RepairMethod.RANDOM:
            importances = list(random.rand(dataset.trainsize))
        else:
            method = IMPORTANCE_METHODS[self.method]
            mc_iterations = MC_ITERATIONS[self.method]
            importance = ShapleyImportance(method=method, utility=utility, mc_iterations=mc_iterations)
            importances = importance.fit(X_train_dirty, y_train_dirty).score(X_val, y_val)
        importance_time_end = process_time_ns()
        self._importance_compute_time = (importance_time_end - importance_time_start) / 1e9
        self.logger.debug("Importance computed in: %s", str(timedelta(seconds=self._importance_compute_time)))
        n_units = dataset.trainsize
        visited_units = np.zeros(n_units, dtype=bool)
        argsorted_importances = np.array(importances).argsort()
        # argsorted_importances = np.ma.array(importances, mask=visited_units).argsort()

        # Run the model to get initial score.
        model.fit(X_train_dirty, y_train_dirty)
        y_pred = model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)

        # Update result table.
        evolution = [[0.0, accuracy, accuracy, 0, 0.0, 0, 0.0]]
        accuracy_start = accuracy

        # Set up progress bar.
        checkpoints = self.checkpoints if self.checkpoints > 0 and self.checkpoints < n_units else n_units
        n_units_per_checkpoint = round(n_units / checkpoints)
        if progress_bar:
            self.progress.start(total=checkpoints, desc="%s Repairs" % str(self))
        # pbar = None if not progress_bar else tqdm(total=dataset.trainsize, desc="%s Repairs" % str(self))

        # Iterate over the repair process.
        # visited_units = np.zeros(dataset.trainsize, dtype=bool)
        for i in range(checkpoints):

            # Determine indices of data examples that should be repaired given the unit with the highest importance.
            unvisited_units = np.invert(visited_units)
            target_units = argsorted_importances[unvisited_units[argsorted_importances]]
            if i + 1 < checkpoints:
                target_units = target_units[:n_units_per_checkpoint]
            target_query = np.zeros(n_units, dtype=int)
            target_query[target_units] = 1
            # target_unit = np.ma.array(importances, mask=visited_units).argmin()
            # target_query = np.eye(1, visited_units.shape[0], target_unit, dtype=int).flatten()
            target_idx = get_indices(provenance, target_query)

            # Repair the data example.
            y_train_dirty[target_idx] = y_train[target_idx]
            visited_units[target_units] = True

            # Run the model.
            model.fit(X_train_dirty, y_train_dirty)
            y_pred = model.predict(X_val)
            accuracy = accuracy_score(y_val, y_pred)

            # Update result table.
            steps_rel = (i + 1) / float(checkpoints)
            repaired = visited_units.sum(dtype=int)
            repaired_rel = repaired / float(n_units)
            discovered = np.logical_and(visited_units, dirty_idx).sum(dtype=int)
            discovered_rel = discovered / dirty_idx.sum(dtype=float)
            evolution.append([steps_rel, accuracy, accuracy, repaired, repaired_rel, discovered, discovered_rel])

            # Recompute if needed.
            if importance is not None and self.method == RepairMethod.KNN_Interactive:
                importances = importance.fit(X_train_dirty, y_train_dirty).score(X_val, y_val)
                argsorted_importances = np.array(importances).argsort()
                # argsorted_importances = np.ma.array(importances, mask=visited_units).argsort()

            # Update progress bar.
            if progress_bar:
                self.progress.update(1)

        # Ensure index column has a label.
        self._evolution = pd.DataFrame(
            evolution,
            columns=[
                "steps_rel",
                "accuracy",
                "accuracy_rel",
                "repaired",
                "repaired_rel",
                "discovered",
                "discovered_rel",
            ],
        )
        self._evolution.index.name = "steps"

        # Fix relative score.
        accuracy_end = accuracy
        accuracy_delta = accuracy_end - accuracy_start
        self._evolution["accuracy_rel"] = self._evolution["accuracy_rel"].apply(
            lambda x: (x - accuracy_start) / accuracy_delta
        )

        # Close progress bar.
        if progress_bar:
            self.progress.close()

    @property
    def completed(self) -> bool:
        dataset = Dataset.datasets[self.dataset]()
        return len(self._evolution) == dataset.trainsize + 1

    @property
    def dataframe(self) -> DataFrame:
        result = self._evolution.assign(
            id=self.id,
            dataset=self.dataset,
            pipeline=self.pipeline,
            method=self.method,
            iteration=self.iteration,
            importance_compute_time=self.importance_compute_time,
        )
        return result

    @property
    def keyword_replacements(self) -> Dict[str, str]:
        return KEYWORD_REPLACEMENTS

    @classmethod
    def is_valid_config(cls, **attributes: Any) -> bool:
        result = True
        if "pipeline" in attributes and "dataset" in attributes:
            dataset = Dataset.datasets[attributes["dataset"]]()
            pipeline = Pipeline.pipelines[attributes["pipeline"]].construct(dataset)
            result = result and dataset.modality in pipeline.modalities
        return result and super().is_valid_config(**attributes)
