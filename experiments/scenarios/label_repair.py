import numpy as np
import pandas as pd

from copy import deepcopy
from datascope.importance.common import SklearnModelUtility, binarize, get_indices
from datascope.importance.shapley import ShapleyImportance, ImportanceMethod
from enum import Enum
from numpy import ndarray
from pandas import DataFrame
from sklearn.metrics import accuracy_score
from time import process_time_ns
from typing import Any, Iterable, Optional

from .base import Scenario, attribute, result
from ..dataset import Dataset
from ..pipelines import Pipeline, get_model, ModelType


class RepairMethod(str, Enum):
    KNN_Single = "shapley-knn-single"
    KNN_Interactive = "shapley-knn-interactive"
    TMC_10 = "shapley-tmc-10"
    TMC_50 = "shapley-tmc-50"
    TMC_100 = "shapley-tmc-100"
    TMC_500 = "shapley-tmc-500"
    RANDOM = "random"


IMPORTANCE_METHODS = {
    RepairMethod.KNN_Single: ImportanceMethod.NEIGHBOR,
    RepairMethod.KNN_Interactive: ImportanceMethod.NEIGHBOR,
    RepairMethod.TMC_10: ImportanceMethod.MONTECARLO,
    RepairMethod.TMC_50: ImportanceMethod.MONTECARLO,
    RepairMethod.TMC_100: ImportanceMethod.MONTECARLO,
    RepairMethod.TMC_500: ImportanceMethod.MONTECARLO,
}


MC_ITERATIONS = {
    RepairMethod.KNN_Single: 0,
    RepairMethod.KNN_Interactive: 0,
    RepairMethod.TMC_10: 10,
    RepairMethod.TMC_50: 50,
    RepairMethod.TMC_100: 100,
    RepairMethod.TMC_500: 500,
}


DEFAULT_DIRTY_RATIO = 0.5
DEFAULT_SEED = 1


class LabelRepairScenario(Scenario, id="label-repair"):
    def __init__(
        self,
        dataset: str,
        pipeline: str,
        method: RepairMethod,
        iteration: int,
        dirty_ratio: float = DEFAULT_DIRTY_RATIO,
        seed: int = DEFAULT_SEED,
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
        dataset = Dataset.datasets[self.dataset]()
        dataset.load()

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
        X_train: ndarray = pipeline.fit_transform(
            dataset.X_train, dataset.y_train
        )  # TODO: Fit the pipeline with dirty data.
        X_train_dirty: ndarray = pipeline.transform(dataset_dirty.X_train)
        y_train, y_val, y_train_dirty = dataset.y_train, dataset.y_val, dataset_dirty.y_train
        X_val: ndarray = pipeline.transform(dataset.X_val)
        assert y_train is not None

        # Reshape datasets if needed.
        if X_train.ndim > 2:
            X_train = X_train.reshape(X_train.shape[0], -1)
            X_train_dirty = X_train_dirty.reshape(X_train_dirty.shape[0], -1)
            X_val = X_val.reshape(X_val.shape[0], -1)

        # Construct binarized provenance matrix.
        provenance = np.expand_dims(np.arange(dataset.trainsize, dtype=int), axis=(1, 2, 3))
        provenance = np.pad(provenance, pad_width=((0, 0), (0, 0), (0, 0), (0, 1)))
        provenance = binarize(provenance)

        # Initialize the model and utility.
        model = get_model(ModelType.LogisticRegression)
        utility = SklearnModelUtility(model, accuracy_score)

        # Compute importance scores and time it.
        time_start = process_time_ns()
        importance: Optional[ShapleyImportance] = None
        importances: Optional[Iterable[float]] = None
        if self.method == RepairMethod.RANDOM:
            importances = list(random.rand(dataset.trainsize))
        else:
            method = IMPORTANCE_METHODS[self.method]
            mc_iterations = MC_ITERATIONS[self.method]
            importance = ShapleyImportance(method=method, utility=utility, mc_iterations=mc_iterations)
            importances = importance.fit(X_train_dirty, y_train_dirty).score(X_val, y_val)
        time_end = process_time_ns()
        self._importance_compute_time = (time_end - time_start) / 1e9

        # Run the model to get initial score.
        model.fit(X_train_dirty, y_train_dirty)
        y_pred = model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)

        # Update result table.
        evolution = [[0.0, accuracy, 0, 0.0, 0, 0.0]]
        accuracy_start = accuracy

        # Set up progress bar.
        if progress_bar:
            self.progress.start(total=dataset.trainsize, desc="%s Repairs" % str(self))
        # pbar = None if not progress_bar else tqdm(total=dataset.trainsize, desc="%s Repairs" % str(self))

        # Iterate over the repair process.
        visited_units = np.zeros(dataset.trainsize, dtype=bool)
        for i in range(dataset.trainsize):

            # Determine indices of data examples that should be repaired given the unit with the highest importance.
            target_unit = np.ma.array(importances, mask=visited_units).argmin()
            target_query = np.eye(1, visited_units.shape[0], target_unit, dtype=int).flatten()
            target_idx = get_indices(provenance, target_query)

            # Repair the data example.
            y_train_dirty[target_idx] = y_train[target_idx]
            visited_units[target_unit] = True

            # Run the model.
            model.fit(X_train_dirty, y_train_dirty)
            y_pred = model.predict(X_val)
            accuracy = accuracy_score(y_val, y_pred)

            # Update result table.
            steps_rel = (i + 1) / float(dataset.trainsize)
            repaired = visited_units.sum(dtype=int)
            repaired_rel = repaired / float(dataset.trainsize)
            discovered = np.logical_and(visited_units, dirty_idx).sum(dtype=int)
            discovered_rel = discovered / dirty_idx.sum(dtype=float)
            evolution.append([steps_rel, accuracy, accuracy, repaired, repaired_rel, discovered, discovered_rel])

            # Recompute if needed.
            if importance is not None and self.method == RepairMethod.KNN_Interactive:
                importances = importance.fit(X_train_dirty, y_train_dirty).score(X_val, y_val)

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

    @classmethod
    def is_valid_config(cls, **attributes: Any) -> bool:
        result = True
        if "pipeline" in attributes and "dataset" in attributes:
            dataset = Dataset.datasets[attributes["dataset"]]()
            pipeline = Pipeline.pipelines[attributes["pipeline"]].construct(dataset)
            result = result and dataset.modality in pipeline.modalities
        return result and super().is_valid_config(**attributes)
