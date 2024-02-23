import numpy as np
import pandas as pd

from datascope.importance.common import (
    Utility,
    SklearnModelAccuracy,
    SklearnModelRocAuc,
)
from datetime import timedelta
from numpy.typing import NDArray
from pandas import DataFrame
from sklearn.preprocessing import FunctionTransformer
from sklearn.utils.multiclass import unique_labels
from time import process_time_ns
from typing import Any, Optional, Dict

from .base import attribute, result, Scenario
from .datascope_scenario import (
    ModelType,
    ModelSpec,
    MODEL_TYPES,
    MODEL_KWARGS,
    DEFAULT_SEED,
    DEFAULT_MODEL,
    KEYWORD_REPLACEMENTS,
    UtilityType,
)
from ..datasets import (
    Dataset,
    DEFAULT_TRAINSIZE,
    DEFAULT_TESTSIZE,
)
from ..pipelines import Pipeline, get_model


class MarginalContributionScenario(Scenario, id="marginal-contribution"):  # type: ignore
    def __init__(
        self,
        dataset: str,
        pipeline: str,
        iteration: int,
        utility: UtilityType = UtilityType.ACCURACY,
        model: ModelSpec = DEFAULT_MODEL,
        seed: int = DEFAULT_SEED,
        trainsize: int = DEFAULT_TRAINSIZE,
        testsize: int = DEFAULT_TESTSIZE,
        evolution: Optional[pd.DataFrame] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self._dataset = dataset
        self._pipeline = pipeline
        self._iteration = iteration
        self._utility = utility
        self._model = model
        self._seed = seed
        self._trainsize = trainsize
        self._testsize = testsize
        self._evolution = pd.DataFrame() if evolution is None else evolution

    @attribute(domain=Dataset.datasets)
    def dataset(self) -> str:
        """Dataset to use for training and validation."""
        return self._dataset

    @attribute(domain=Pipeline.pipelines)
    def pipeline(self) -> str:
        """Pipeline to use for feature extraction."""
        return self._pipeline

    @attribute(domain=[None])
    def utility(self) -> UtilityType:
        """Utility to measure."""
        return self._utility

    @attribute(domain=[None])
    def model(self) -> ModelSpec:
        """Model used to make predictions."""
        return self._model

    @attribute(domain=range(10))
    def iteration(self) -> int:
        """The ordinal number of the experiment repetition. Also serves as the random seed."""
        return self._iteration

    @attribute
    def trainsize(self) -> int:
        """The size of the training dataset to use. The value 0 means maximal value."""
        return self._trainsize

    @attribute
    def testsize(self) -> int:
        """The size of the test dataset to use. The value 0 means maximal value."""
        return self._testsize

    @result
    def evolution(self) -> DataFrame:
        """The evolution of the experimental parameters."""
        return self._evolution

    @property
    def completed(self) -> bool:
        return len(self._evolution.index) > 0 and self._evolution.index[-1] == self._trainsize - 1

    @property
    def dataframe(self) -> DataFrame:
        result = self._evolution.assign(
            dataset=self.dataset,
            pipeline=self.pipeline,
            model=str(self.model),
            utility=str(self.utility),
            iteration=self.iteration,
        )
        return result

    @property
    def keyword_replacements(self) -> Dict[str, str]:
        return {**KEYWORD_REPLACEMENTS, **Pipeline.summaries, **Dataset.summaries}

    @classmethod
    def is_valid_config(cls, **attributes: Any) -> bool:
        result = True
        if "pipeline" in attributes and "dataset" in attributes:
            dataset = Dataset.datasets[attributes["dataset"]]
            pipeline = Pipeline.pipelines[attributes["pipeline"]](steps=[("hack", FunctionTransformer())])
            result = result and any(issubclass(dataset, modality) for modality in pipeline.modalities)
        if "utility" in attributes:
            result = result and attributes["utility"] in [UtilityType.ACCURACY, UtilityType.ROC_AUC, UtilityType.EQODDS]
        return result and super().is_valid_config(**attributes)

    def _run(self, progress_bar: bool = True, **kwargs: Any) -> None:
        # Load dataset.
        seed = self._seed + self._iteration
        random = np.random.RandomState(seed=seed)
        dataset = Dataset.datasets[self.dataset](
            trainsize=self.trainsize, valsize=self.testsize, testsize=self.testsize, seed=seed
        )
        dataset.load()
        self.logger.debug(
            "Dataset loaded (dataset=%s, trainsize=%d, testsize=%d).",
            self.dataset,
            dataset.trainsize,
            dataset.testsize,
        )

        # Construct the utilities.
        model_type = MODEL_TYPES[self.model]
        model_kwargs = MODEL_KWARGS[self.model]
        model = get_model(model_type, **model_kwargs)
        utility: Utility
        if self._utility == UtilityType.ACCURACY:
            utility = SklearnModelAccuracy(model)
        elif self._utility == UtilityType.ROC_AUC:
            utility = SklearnModelRocAuc(model)
        else:
            raise ValueError("Utility '%s' unsupported." % self._utility)

        # Run the pipeline on the dataset.
        pipeline_runtime_start = process_time_ns()
        pipeline = Pipeline.pipelines[self.pipeline].construct(dataset)
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
        if model_type == ModelType.KNeighbors:
            cardinalities_lower = max(cardinalities_lower, model_kwargs.get("n_neighbors", 0))
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
