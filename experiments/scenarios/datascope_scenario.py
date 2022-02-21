import pandas as pd

from datascope.importance.shapley import ImportanceMethod
from enum import Enum
from pandas import DataFrame
from typing import Any, Optional, Dict

from .base import Scenario, attribute, result
from ..dataset import Dataset, DEFAULT_TRAINSIZE, DEFAULT_VALSIZE
from ..pipelines import Pipeline


class RepairMethod(str, Enum):
    KNN_Single = "shapley-knn-single"
    KNN_Interactive = "shapley-knn-interactive"
    TMC_1 = "shapley-tmc-001"
    TMC_5 = "shapley-tmc-005"
    TMC_10 = "shapley-tmc-010"
    TMC_50 = "shapley-tmc-050"
    TMC_100 = "shapley-tmc-100"
    TMC_500 = "shapley-tmc-500"
    TMC_PIPE_1 = "shapley-tmc-pipe-001"
    TMC_PIPE_5 = "shapley-tmc-pipe-005"
    TMC_PIPE_10 = "shapley-tmc-pipe-010"
    TMC_PIPE_50 = "shapley-tmc-pipe-050"
    TMC_PIPE_100 = "shapley-tmc-pipe-100"
    TMC_PIPE_500 = "shapley-tmc-pipe-500"
    RANDOM = "random"

    @staticmethod
    def is_pipe(method: "RepairMethod") -> bool:
        return method in [
            RepairMethod.TMC_PIPE_1,
            RepairMethod.TMC_PIPE_5,
            RepairMethod.TMC_PIPE_10,
            RepairMethod.TMC_PIPE_50,
            RepairMethod.TMC_PIPE_100,
            RepairMethod.TMC_PIPE_500,
        ]


class UtilityType(str, Enum):
    ACCURACY = "acc"
    EQODDS = "eqodds"
    EQODDS_AND_ACCURACY = "eqodds-acc"


IMPORTANCE_METHODS = {
    RepairMethod.KNN_Single: ImportanceMethod.NEIGHBOR,
    RepairMethod.KNN_Interactive: ImportanceMethod.NEIGHBOR,
    RepairMethod.TMC_1: ImportanceMethod.MONTECARLO,
    RepairMethod.TMC_5: ImportanceMethod.MONTECARLO,
    RepairMethod.TMC_10: ImportanceMethod.MONTECARLO,
    RepairMethod.TMC_50: ImportanceMethod.MONTECARLO,
    RepairMethod.TMC_100: ImportanceMethod.MONTECARLO,
    RepairMethod.TMC_500: ImportanceMethod.MONTECARLO,
    RepairMethod.TMC_PIPE_1: ImportanceMethod.MONTECARLO,
    RepairMethod.TMC_PIPE_5: ImportanceMethod.MONTECARLO,
    RepairMethod.TMC_PIPE_10: ImportanceMethod.MONTECARLO,
    RepairMethod.TMC_PIPE_50: ImportanceMethod.MONTECARLO,
    RepairMethod.TMC_PIPE_100: ImportanceMethod.MONTECARLO,
    RepairMethod.TMC_PIPE_500: ImportanceMethod.MONTECARLO,
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
    RepairMethod.TMC_PIPE_1: 1,
    RepairMethod.TMC_PIPE_5: 5,
    RepairMethod.TMC_PIPE_10: 10,
    RepairMethod.TMC_PIPE_50: 50,
    RepairMethod.TMC_PIPE_100: 100,
    RepairMethod.TMC_PIPE_500: 500,
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
    "shapley-tmc-pipe-001": "Shapley TMC Pipe x1",
    "shapley-tmc-pipe-005": "Shapley TMC Pipe x5",
    "shapley-tmc-pipe-010": "Shapley TMC Pipe x10",
    "shapley-tmc-pipe-050": "Shapley TMC Pipe x50",
    "shapley-tmc-pipe-100": "Shapley TMC Pipe x100",
    "shapley-tmc-pipe-500": "Shapley TMC Pipe x500",
    "eqodds": "Equalized Odds Difference",
}

DEFAULT_SEED = 1
DEFAULT_CHECKPOINTS = 100
DEFAULT_TIMEOUT = 3600


class DatascopeScenario(Scenario):
    def __init__(
        self,
        dataset: str,
        pipeline: str,
        method: RepairMethod,
        utility: UtilityType,
        iteration: int,
        seed: int = DEFAULT_SEED,
        trainsize: int = DEFAULT_TRAINSIZE,
        valsize: int = DEFAULT_VALSIZE,
        timeout: int = DEFAULT_TIMEOUT,
        checkpoints: int = DEFAULT_CHECKPOINTS,
        evolution: Optional[pd.DataFrame] = None,
        importance_compute_time: Optional[float] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self._dataset = dataset
        self._pipeline = pipeline
        self._method = method
        self._utility = utility
        self._iteration = iteration
        self._seed = seed
        self._trainsize = trainsize
        self._valsize = valsize
        self._timeout = timeout
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

    @attribute
    def utility(self) -> UtilityType:
        """The utility to use for importance computation."""
        return self._utility

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
    def timeout(self) -> int:
        """The maximum time in seconds that a Monte-Carlo importance method is allowed to run."""
        return self._timeout

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
            utility=self.utility,
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
