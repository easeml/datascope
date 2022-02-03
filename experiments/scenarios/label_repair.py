import numpy as np
import pandas as pd

from copy import deepcopy
from datascope.importance.common import SklearnModelUtility, binarize, get_indices
from datascope.importance.shapley import ShapleyImportance, ImportanceMethod
from enum import Enum

from .base import Scenario, attribute, result
from ..datasets import Dataset
from ..pipelines import Pipeline, get_model, ModelType

from pandas import DataFrame
from sklearn.metrics import accuracy_score
from typing import Any, Iterable, Optional


class RepairMethod(str, Enum):
    KNN_Single = "shapley-knn-single"
    KNN_Interactive = "shapley-knn-interactive"
    TMC = "shapley-tmc"
    RANDOM = "random"


IMPORTANCE_METHODS = {
    RepairMethod.KNN_Single: ImportanceMethod.NEIGHBOR,
    RepairMethod.KNN_Interactive: ImportanceMethod.NEIGHBOR,
    RepairMethod.TMC: ImportanceMethod.MONTECARLO,
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
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self._dataset = dataset
        self._pipeline = pipeline
        self._method = method
        self._iteration = iteration
        self._dirty_ratio = dirty_ratio
        self._seed = seed
        self._evolution = pd.DataFrame()

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

    def _run(self, progress_bar: bool = True, **kwargs: Any) -> None:

        # Load dataset.
        dataset = Dataset.datasets[self.dataset]()
        dataset.load()

        # Create the dirty dataset and apply the data corruption.
        dataset_dirty = deepcopy(dataset)
        random = np.random.RandomState(seed=self._seed + self._iteration)
        dirty_idx = random.choice(a=[False, True], size=(dataset_dirty.trainsize))
        assert dataset_dirty.y_train is not None
        dataset_dirty.y_train[dirty_idx] = 1 - dataset_dirty.y_train[dirty_idx]

        # Load the pipeline and process the data.
        pipeline_class = Pipeline.pipelines[self.pipeline]
        pipeline = pipeline_class.construct(dataset)
        X_train = pipeline.fit_transform(dataset.X_train, dataset.y_train)  # TODO: Fit the pipeline with dirty data.
        X_train_dirty = pipeline.transform(dataset_dirty.X_train)
        y_train, y_val, y_train_dirty = dataset.y_train, dataset.y_val, dataset_dirty.y_train
        X_val = pipeline.transform(dataset.X_val)
        assert y_train is not None

        # Construct binarized provenance matrix.
        provenance = np.expand_dims(np.arange(dataset.trainsize, dtype=int), axis=(1, 2, 3))
        provenance = np.pad(provenance, pad_width=((0, 0), (0, 0), (0, 0), (0, 1)))
        provenance = binarize(provenance)

        # Initialize the model and utility.
        model = get_model(ModelType.LogisticRegression)
        utility = SklearnModelUtility(model, accuracy_score)

        # Compute importance scores.
        importance: Optional[ShapleyImportance] = None
        importances: Optional[Iterable[float]] = None
        if self.method == RepairMethod.RANDOM:
            importances = list(random.rand(dataset.trainsize))
        else:
            method = IMPORTANCE_METHODS[self.method]
            importance = ShapleyImportance(method=method, utility=utility)
            importances = importance.fit(X_train, y_train).score(X_val, y_val)

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
                importances = importance.fit(X_train, y_train).score(X_val, y_val)

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

    def completed(self) -> bool:
        dataset = Dataset.datasets[self.dataset]()
        return len(self._evolution) == dataset.trainsize + 1

    def dataframe(self) -> DataFrame:
        result = self._evolution.assign(
            id=self.id, dataset=self.dataset, pipeline=self.pipeline, method=self.method, iteration=self.iteration
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
