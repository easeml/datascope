import numpy as np

from datascope.importance import SklearnModelAccuracy, ShapleyImportance
from datetime import timedelta
from methodtools import lru_cache
from pandas import DataFrame
from time import process_time_ns
from typing import Any, Optional, Dict, Type

from ..bench import Scenario, attribute, result
from .data_repair_scenario import (
    RepairMethod,
    IMPORTANCE_METHODS,
    MC_ITERATIONS,
    DEFAULT_SEED,
    DEFAULT_MODEL,
    DEFAULT_MC_TIMEOUT,
    KEYWORD_REPLACEMENTS,
    UtilityType,
)
from ..datasets import Dataset, RandomDataset
from ..pipelines import Pipeline, Model

DEFAULT_DATASET = RandomDataset()


class ComputeTimeScenario(Scenario, id="compute-time"):
    def __init__(
        self,
        pipeline: Pipeline,
        method: RepairMethod,
        dataset: Dataset = DEFAULT_DATASET,
        model: Model = DEFAULT_MODEL,
        seed: int = DEFAULT_SEED,
        mc_timeout: int = DEFAULT_MC_TIMEOUT,
        importance_cputime: Optional[float] = None,
        iteration: Optional[int] = None,  # Iteration became seed. Keeping this for back compatibility reasons.
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self._dataset = dataset
        self._pipeline = pipeline
        self._method = method
        self._iteration = iteration
        self._model = model
        self._seed = seed
        if iteration is not None:
            self._seed += iteration
        self._mc_timeout = mc_timeout
        self._importance_cputime: Optional[float] = importance_cputime

    @classmethod
    def is_valid_config(cls, **attributes: Any) -> bool:
        result = True
        if "utility" in attributes:
            result = attributes["utility"] == UtilityType.ACCURACY
        if "method" in attributes:
            result = result and not RepairMethod.is_tmc_nonpipe(attributes["method"])
        return result and super().is_valid_config(**attributes)

    @attribute
    def dataset(self) -> Dataset:
        """Dataset to use for training and validation."""
        return self._dataset

    @attribute
    def pipeline(self) -> Pipeline:
        """Pipeline to use for feature extraction."""
        return self._pipeline

    @attribute
    def method(self) -> RepairMethod:
        """Method used to perform data repairs."""
        return self._method

    @attribute
    def model(self) -> Model:
        """Model used to make predictions."""
        return self._model

    @attribute(domain=range(10))
    def seed(self) -> int:
        """The random seed that applies to all pseudo-random processes in the scenario."""
        return self._seed

    @attribute
    def mc_timeout(self) -> int:
        """The maximum time in seconds that a Monte-Carlo importance method is allowed to run."""
        return self._mc_timeout

    @result
    def importance_cputime(self) -> Optional[float]:
        """The time it takes to compute importance."""
        return self._importance_cputime

    @property
    def completed(self) -> bool:
        return self.importance_cputime is not None

    @property
    def dataframe(self) -> DataFrame:
        attributes = self._flatten_attributes(self.attributes)
        return DataFrame(attributes)

    @lru_cache(maxsize=1)
    @classmethod
    def get_keyword_replacements(cls: Type["Scenario"]) -> Dict[str, str]:
        return {
            **KEYWORD_REPLACEMENTS,
            **Dataset.get_keyword_replacements(),
            **Pipeline.get_keyword_replacements(),
            **Model.get_keyword_replacements(),
        }

    def _run(self, progress_bar: bool = True, **kwargs: Any) -> None:
        # Load dataset.
        dataset = self.dataset.load()
        self.logger.debug("Loaded dataset %s.", dataset)

        # Load the pipeline and process the data.
        pipeline = self.pipeline.construct(dataset)

        # Initialize the model and utility.
        model = self.model.construct(dataset)
        utility = SklearnModelAccuracy(model)

        # Compute importance scores and time it.
        importance_time_start = process_time_ns()
        n_units = self.dataset.provenance.num_units
        importance: Optional[ShapleyImportance] = None
        random = np.random.RandomState(seed=self.seed + self._iteration)
        if self.method == RepairMethod.RANDOM:
            list(random.rand(n_units))
        else:
            method = IMPORTANCE_METHODS[self.method]
            mc_iterations = MC_ITERATIONS[self.method]
            mc_preextract = RepairMethod.is_tmc_nonpipe(self.method)
            importance = ShapleyImportance(
                method=method,
                utility=utility,
                pipeline=pipeline,
                mc_iterations=mc_iterations,
                mc_timeout=self.mc_timeout,
                mc_preextract=mc_preextract,
            )
            importance.fit(dataset.X_train, dataset.y_train, dataset.metadata_train).score(
                dataset.X_val, dataset.y_val, dataset.metadata_val
            )
        importance_time_end = process_time_ns()
        self._importance_cputime = (importance_time_end - importance_time_start) / 1e9
        self.logger.debug("Importance computed in: %s", str(timedelta(seconds=self._importance_cputime)))
