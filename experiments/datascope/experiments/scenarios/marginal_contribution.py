import numpy as np
import pandas as pd

from datascope.importance import (
    Utility,
    SklearnModelAccuracy,
    SklearnModelRocAuc,
)
from datetime import timedelta
from methodtools import lru_cache
from numpy.typing import NDArray
from pandas import DataFrame
from sklearn.utils.multiclass import unique_labels
from time import process_time_ns
from typing import Any, Optional, Dict, Type

from ..bench import attribute, result, Scenario
from .data_repair_scenario import (
    DEFAULT_SEED,
    DEFAULT_MODEL,
    KEYWORD_REPLACEMENTS,
    UtilityType,
)
from ..datasets import Dataset
from ..pipelines import Pipeline, Model, KNearestNeighborsModel

KEYWORD_REPLACEMENTS = {
    **KEYWORD_REPLACEMENTS,
    "cardinality": "Cardinality",
    "score_without": "Score without",
    "score_with": "Score with",
    "score_marginal": "Marginal score",
}


class MarginalContributionScenario(Scenario, id="marginal-contribution"):  # type: ignore
    def __init__(
        self,
        dataset: Dataset,
        pipeline: Pipeline,
        utility: UtilityType = UtilityType.ACCURACY,
        model: Model = DEFAULT_MODEL,
        seed: int = DEFAULT_SEED,
        evolution: Optional[pd.DataFrame] = None,
        iteration: Optional[int] = None,  # Iteration became seed. Keeping this for back compatibility reasons.
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self._dataset = dataset
        self._pipeline = pipeline
        self._utility = utility
        self._model = model
        self._seed = seed
        if iteration is not None:
            self._seed += iteration
        self._evolution = pd.DataFrame() if evolution is None else evolution

    @attribute
    def dataset(self) -> Dataset:
        """Dataset to use for training and validation."""
        return self._dataset

    @attribute
    def pipeline(self) -> Pipeline:
        """Pipeline to use for feature extraction."""
        return self._pipeline

    @attribute(domain=[None])
    def utility(self) -> UtilityType:
        """Utility to measure."""
        return self._utility

    @attribute
    def model(self) -> Model:
        """Model used to make predictions."""
        return self._model

    @attribute(domain=range(10))
    def seed(self) -> int:
        """The random seed that applies to all pseudo-random processes in the scenario."""
        return self._seed

    @result
    def evolution(self) -> DataFrame:
        """The evolution of the experimental parameters."""
        return self._evolution

    @property
    def completed(self) -> bool:
        return len(self._evolution.index) > 0 and self._evolution.index[-1] == self.dataset.trainsize - 1

    @property
    def dataframe(self) -> DataFrame:
        attributes = self._flatten_attributes(self.attributes)
        result = self._evolution.assign(**attributes)
        return result

    @lru_cache(maxsize=1)
    @classmethod
    def get_keyword_replacements(cls: Type["Scenario"]) -> Dict[str, str]:
        return {
            **KEYWORD_REPLACEMENTS,
            **Dataset.get_keyword_replacements(),
            **Pipeline.get_keyword_replacements(),
            **Model.get_keyword_replacements(),
        }

    @classmethod
    def is_valid_config(cls, **attributes: Any) -> bool:
        result = True
        dataset: Optional[Dataset] = attributes.get("dataset", None)
        pipeline: Optional[Pipeline] = attributes.get("pipeline", None)
        utility: UtilityType = attributes.get("utility", None)

        if pipeline is not None and dataset is not None:
            result = result and any(isinstance(dataset, modality) for modality in pipeline.modalities)
        if utility is not None:
            result = result and utility in [UtilityType.ACCURACY, UtilityType.ROC_AUC, UtilityType.EQODDS]
        return result and super().is_valid_config(**attributes)

    def _run(self, progress_bar: bool = True, **kwargs: Any) -> None:
        # Load dataset.
        random = np.random.RandomState(seed=self.seed)
        dataset = self.dataset.load()
        self.logger.debug("Loaded dataset %s.", dataset)

        # Construct the utilities.
        model = self.model.construct(dataset)
        utility: Utility
        if self._utility == UtilityType.ACCURACY:
            utility = SklearnModelAccuracy(model)
        elif self._utility == UtilityType.ROC_AUC:
            utility = SklearnModelRocAuc(model)
        else:
            raise ValueError("Utility '%s' unsupported." % self._utility)

        # Run the pipeline on the dataset.
        pipeline_runtime_start = process_time_ns()
        pipeline = self.pipeline.construct(dataset)
        dataset_processed = dataset.apply(pipeline)
        pipeline_runtime_end = process_time_ns()
        self._pipeline_runtime = (pipeline_runtime_end - pipeline_runtime_start) / 1e9
        self.logger.info(
            "Features extracted (pipeline=%s, runtime=%s).",
            self.pipeline,
            str(timedelta(seconds=self._pipeline_runtime)),
        )

        # Construct a permutation such that each of the first M elements has one of the M class labels.
        labels = unique_labels(dataset.y_train)
        permutation: NDArray[np.int_] = random.permutation(np.arange(dataset.trainsize, dtype=np.int_))
        for i, label in enumerate(labels):
            j = np.where(dataset.y_train[permutation] == label)[0][0]
            permutation[[i, j]] = permutation[[j, i]]

        # Determine the upper and lower bounds of cardinalities.
        cardinalities_lower, cardinalities_upper = len(labels), dataset.trainsize - 1
        if isinstance(self.model, KNearestNeighborsModel):
            cardinalities_lower = max(cardinalities_lower, self.model.n_neighbors)
        if progress_bar:
            self.progress.start(total=100, desc="(id=%s) Progress" % self.id)
        evolution = []
        progress_points = list(np.linspace(cardinalities_lower, cardinalities_upper, 101, dtype=int))[1:]

        for cardinality in range(cardinalities_lower, cardinalities_upper):
            idx_without = permutation[:cardinality]
            metadata_train_without = (
                dataset_processed.metadata_train.iloc[idx_without] if dataset_processed.metadata_train else None
            )
            utility_result_without = utility(
                dataset_processed.X_train[idx_without],
                dataset_processed.y_train[idx_without],
                dataset_processed.X_test,
                dataset_processed.y_test,
                metadata_train=metadata_train_without,
                metadata_test=dataset_processed.metadata_test,
            )
            idx_with = np.concatenate((permutation[:cardinality], permutation[-1:]), axis=0)
            metadata_train_with = (
                dataset_processed.metadata_train.iloc[idx_with] if dataset_processed.metadata_train else None
            )
            utility_result_with = utility(
                dataset_processed.X_train[idx_with],
                dataset_processed.y_train[idx_with],
                dataset_processed.X_test,
                dataset_processed.y_test,
                metadata_train=metadata_train_with,
                metadata_test=dataset_processed.metadata_test,
            )
            evolution.append(
                [
                    cardinality,
                    utility_result_without.score,
                    utility_result_with.score,
                    utility_result_with.score - utility_result_without.score,
                ]
            )

            # Update progress bar.
            if progress_bar and cardinality >= progress_points[0]:
                self.progress.update(1)
                progress_points.pop(0)

        # Assemble evolution as a data frame.
        self._evolution = pd.DataFrame(
            evolution, columns=["cardinality", "score_without", "score_with", "score_marginal"]
        )
        self._evolution.set_index("cardinality", inplace=True)

        # Close progress bar.
        if progress_bar:
            self.progress.close()
