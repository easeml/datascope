import numpy as np

from datascope.importance.common import SklearnModelAccuracy
from datascope.importance.shapley import ShapleyImportance
from datetime import timedelta
from pandas import DataFrame
from time import process_time_ns
from typing import Any, Optional, Dict

from .base import Scenario, attribute, result
from .datascope_scenario import (
    RepairMethod,
    IMPORTANCE_METHODS,
    MC_ITERATIONS,
    DEFAULT_SEED,
    DEFAULT_MODEL,
    DEFAULT_TIMEOUT,
    KEYWORD_REPLACEMENTS,
    UtilityType,
)
from ..datasets import Dataset, DEFAULT_TRAINSIZE, DEFAULT_VALSIZE, DEFAULT_NUMFEATURES
from ..pipelines import Pipeline, ModelType, get_model

DEFAULT_DATASET = "random"
KEYWORD_REPLACEMENTS = {
    **KEYWORD_REPLACEMENTS,
    **{"trainsize": "Training Set Size", "valsize": "Validation Set Size", "numfeatures": "Number of Features"},
}


class ComputeTimeScenario(Scenario, id="compute-time"):
    def __init__(
        self,
        pipeline: str,
        method: RepairMethod,
        iteration: int,
        dataset: str = DEFAULT_DATASET,
        model: ModelType = DEFAULT_MODEL,
        seed: int = DEFAULT_SEED,
        trainsize: int = DEFAULT_TRAINSIZE,
        valsize: int = DEFAULT_VALSIZE,
        numfeatures: int = DEFAULT_NUMFEATURES,
        timeout: int = DEFAULT_TIMEOUT,
        importance_cputime: Optional[float] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self._dataset = dataset
        self._pipeline = pipeline
        self._method = method
        self._iteration = iteration
        self._model = model
        self._seed = seed
        self._trainsize = trainsize
        self._valsize = valsize
        self._numfeatures = numfeatures
        self._timeout = timeout
        self._importance_cputime: Optional[float] = importance_cputime

    @classmethod
    def is_valid_config(cls, **attributes: Any) -> bool:
        result = True
        if "utility" in attributes:
            result = attributes["utility"] == UtilityType.ACCURACY
        if "method" in attributes:
            result = result and not RepairMethod.is_tmc_nonpipe(attributes["method"])
        return result and super().is_valid_config(**attributes)

    @attribute(domain=[DEFAULT_DATASET])
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

    @attribute(domain=[None])
    def model(self) -> ModelType:
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
    def valsize(self) -> int:
        """The size of the validation dataset to use. The value 0 means maximal value."""
        return self._valsize

    @attribute
    def numfeatures(self) -> int:
        """The number of features to have in the dataset when it is generated."""
        return self._numfeatures

    @attribute
    def timeout(self) -> int:
        """The maximum time in seconds that a Monte-Carlo importance method is allowed to run."""
        return self._timeout

    @result
    def importance_cputime(self) -> Optional[float]:
        """The time it takes to compute importance."""
        return self._importance_cputime

    @property
    def completed(self) -> bool:
        return self.importance_cputime is not None

    @property
    def dataframe(self) -> DataFrame:
        return DataFrame(
            dict(
                dataset=[self.dataset],
                pipeline=[self.pipeline],
                model=[self.model],
                method=[self.method],
                iteration=[self.iteration],
                trainsize=[self.trainsize],
                valsize=[self.valsize],
                numfeatures=[self.numfeatures],
                timeout=[self.timeout],
                importance_cputime=[self.importance_cputime],
            )
        )

    @property
    def keyword_replacements(self) -> Dict[str, str]:
        return {**KEYWORD_REPLACEMENTS, **Pipeline.summaries}

    def _run(self, progress_bar: bool = True, **kwargs: Any) -> None:

        # Load dataset.
        seed = self._seed + self._iteration
        dataset = Dataset.datasets[self.dataset](
            trainsize=self.trainsize, valsize=self.valsize, numfeatures=self.numfeatures, seed=seed
        )
        dataset.load()
        self.logger.debug(
            "Dataset '%s' loaded (trainsize=%d, valsize=%d).", self.dataset, dataset.trainsize, dataset.valsize
        )

        # Load the pipeline and process the data.
        pipeline = Pipeline.pipelines[self.pipeline].construct(dataset)

        # Initialize the model and utility.
        model = get_model(self.model)
        utility = SklearnModelAccuracy(model)

        # Compute importance scores and time it.
        importance_time_start = process_time_ns()
        n_units = dataset.units.shape[0]
        importance: Optional[ShapleyImportance] = None
        random = np.random.RandomState(seed=self._seed + self._iteration)
        if self.method == RepairMethod.RANDOM:
            list(random.rand(n_units))
        else:
            method = IMPORTANCE_METHODS[self.method]
            mc_iterations = MC_ITERATIONS[self.method]
            mc_preextract = RepairMethod.is_tmc_nonpipe(self.method)
            provenance = np.array(np.nan)  # TODO: Remove this hack.
            importance = ShapleyImportance(
                method=method,
                utility=utility,
                pipeline=pipeline,
                mc_iterations=mc_iterations,
                mc_timeout=self.timeout,
                mc_preextract=mc_preextract,
            )
            importance.fit(dataset.X_train, dataset.y_train, provenance=provenance).score(dataset.X_val, dataset.y_val)
        importance_time_end = process_time_ns()
        self._importance_cputime = (importance_time_end - importance_time_start) / 1e9
        self.logger.debug("Importance computed in: %s", str(timedelta(seconds=self._importance_cputime)))
