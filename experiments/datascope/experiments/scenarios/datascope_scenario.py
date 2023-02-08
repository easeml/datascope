import pandas as pd

from datascope.importance.shapley import ImportanceMethod, DEFAULT_MC_TIMEOUT, DEFAULT_MC_TOLERANCE, DEFAULT_NN_K
from enum import Enum
from pandas import DataFrame
from sklearn.preprocessing import FunctionTransformer
from typing import Any, Optional, Dict

from .base import Scenario, attribute, result
from ..datasets import (
    Dataset,
    BiasMethod,
    DEFAULT_TRAINSIZE,
    DEFAULT_VALSIZE,
    DEFAULT_TESTSIZE,
    DEFAULT_BIAS_METHOD,
    KEYWORD_REPLACEMENTS as DATASET_KEYWORD_REPLACEMENTS,
)
from ..pipelines import Pipeline, ModelType, MODEL_KEYWORD_REPLACEMENTS


class ModelSpec(str, Enum):
    LogisticRegression = "logreg"
    RandomForest = "randf"
    KNeighbors = "knn"
    KNeighbors_1 = "knn-1"
    KNeighbors_5 = "knn-5"
    KNeighbors_10 = "knn-10"
    KNeighbors_50 = "knn-50"
    KNeighbors_100 = "knn-100"
    SVM = "svm"
    LinearSVM = "linsvm"
    GaussianProcess = "gp"
    NaiveBayes = "nb"
    NeuralNetwork = "nn"
    XGBoost = "xgb"


MODEL_TYPES = {
    ModelSpec.LogisticRegression: ModelType.LogisticRegression,
    ModelSpec.RandomForest: ModelType.RandomForest,
    ModelSpec.KNeighbors: ModelType.KNeighbors,
    ModelSpec.KNeighbors_1: ModelType.KNeighbors,
    ModelSpec.KNeighbors_5: ModelType.KNeighbors,
    ModelSpec.KNeighbors_10: ModelType.KNeighbors,
    ModelSpec.SVM: ModelType.SVM,
    ModelSpec.LinearSVM: ModelType.LinearSVM,
    ModelSpec.GaussianProcess: ModelType.GaussianProcess,
    ModelSpec.NaiveBayes: ModelType.NaiveBayes,
    ModelSpec.NeuralNetwork: ModelType.NeuralNetwork,
    ModelSpec.XGBoost: ModelType.XGBoost,
}

MODEL_KWARGS: Dict[ModelSpec, Dict[str, Any]] = {
    ModelSpec.LogisticRegression: {},
    ModelSpec.RandomForest: {},
    ModelSpec.KNeighbors: {},
    ModelSpec.KNeighbors_1: {"n_neighbors": 1},
    ModelSpec.KNeighbors_5: {"n_neighbors": 5},
    ModelSpec.KNeighbors_10: {"n_neighbors": 10},
    ModelSpec.KNeighbors_50: {"n_neighbors": 50},
    ModelSpec.KNeighbors_100: {"n_neighbors": 100},
    ModelSpec.SVM: {},
    ModelSpec.LinearSVM: {},
    ModelSpec.GaussianProcess: {},
    ModelSpec.NaiveBayes: {},
    ModelSpec.NeuralNetwork: {},
    ModelSpec.XGBoost: {},
}


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

    @staticmethod
    def is_tmc(method: "RepairMethod") -> bool:
        return method in [
            RepairMethod.TMC_1,
            RepairMethod.TMC_5,
            RepairMethod.TMC_10,
            RepairMethod.TMC_50,
            RepairMethod.TMC_100,
            RepairMethod.TMC_500,
            RepairMethod.TMC_PIPE_1,
            RepairMethod.TMC_PIPE_5,
            RepairMethod.TMC_PIPE_10,
            RepairMethod.TMC_PIPE_50,
            RepairMethod.TMC_PIPE_100,
            RepairMethod.TMC_PIPE_500,
        ]

    @staticmethod
    def is_tmc_nonpipe(method: "RepairMethod") -> bool:
        return method in [
            RepairMethod.TMC_1,
            RepairMethod.TMC_5,
            RepairMethod.TMC_10,
            RepairMethod.TMC_50,
            RepairMethod.TMC_100,
            RepairMethod.TMC_500,
        ]


class RepairGoal(str, Enum):
    FAIRNESS = "fairness"
    ACCURACY = "accuracy"


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
    "shapley-tmc-001": "Shapley TMC+PP x1",
    "shapley-tmc-005": "Shapley TMC+PP x5",
    "shapley-tmc-010": "Shapley TMC+PP x10",
    "shapley-tmc-050": "Shapley TMC+PP x50",
    "shapley-tmc-100": "Shapley TMC+PP x100",
    "shapley-tmc-500": "Shapley TMC+PP x500",
    "shapley-tmc-pipe-001": "Shapley TMC x1",
    "shapley-tmc-pipe-005": "Shapley TMC x5",
    "shapley-tmc-pipe-010": "Shapley TMC x10",
    "shapley-tmc-pipe-050": "Shapley TMC x50",
    "shapley-tmc-pipe-100": "Shapley TMC x100",
    "shapley-tmc-pipe-500": "Shapley TMC x500",
    "knn": "K-Nearest Neighbor (K=1)",
    "knn-1": "K-Nearest Neighbor (K=1)",
    "knn-5": "K-Nearest Neighbor (K=5)",
    "knn-10": "K-Nearest Neighbor (K=10)",
    "knn-50": "K-Nearest Neighbor (K=50)",
    "knn-100": "K-Nearest Neighbor (K=100)",
    "eqodds": "Equalized Odds Difference",
    "importance_cputime": "Compute Time [s]",
    "steps": "Repair Steps Taken",
    "steps_rel": "Relative Repair Steps Taken",
    "acc": "Accuracy",
    "eqodds-acc": "Accuracy + Equalized Odds Difference",
}
KEYWORD_REPLACEMENTS.update(DATASET_KEYWORD_REPLACEMENTS)
KEYWORD_REPLACEMENTS.update(MODEL_KEYWORD_REPLACEMENTS)

DEFAULT_SEED = 1
DEFAULT_CHECKPOINTS = 100
DEFAULT_PROVIDERS = 0
DEFAULT_MODEL = ModelSpec.LogisticRegression
DEFAULT_REPAIR_GOAL = RepairGoal.ACCURACY
DEFAULT_TRAIN_BIAS = 0.0
DEFAULT_VAL_BIAS = 0.0


class DatascopeScenario(Scenario, abstract=True):
    def __init__(
        self,
        dataset: str,
        pipeline: str,
        method: RepairMethod,
        utility: UtilityType,
        iteration: int,
        model: ModelSpec = DEFAULT_MODEL,
        trainbias: float = DEFAULT_TRAIN_BIAS,
        valbias: float = DEFAULT_VAL_BIAS,
        biasmethod: BiasMethod = DEFAULT_BIAS_METHOD,
        seed: int = DEFAULT_SEED,
        trainsize: int = DEFAULT_TRAINSIZE,
        valsize: int = DEFAULT_VALSIZE,
        testsize: int = DEFAULT_TESTSIZE,
        mc_timeout: int = DEFAULT_MC_TIMEOUT,
        mc_tolerance: float = DEFAULT_MC_TOLERANCE,
        nn_k: int = DEFAULT_NN_K,
        checkpoints: int = DEFAULT_CHECKPOINTS,
        providers: int = DEFAULT_PROVIDERS,
        repairgoal: RepairGoal = DEFAULT_REPAIR_GOAL,
        evolution: Optional[pd.DataFrame] = None,
        importance_cputime: Optional[float] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self._dataset = dataset
        self._pipeline = pipeline
        self._method = method
        self._utility = utility
        self._iteration = iteration
        self._model = model
        self._trainbias = trainbias
        self._valbias = valbias
        self._biasmethod = biasmethod
        self._seed = seed
        self._trainsize = trainsize
        self._valsize = valsize
        self._testsize = testsize
        self._mc_timeout = mc_timeout
        self._mc_tolerance = mc_tolerance
        self._nn_k = nn_k
        self._checkpoints = checkpoints
        self._providers = providers
        self._repairgoal = repairgoal
        self._evolution = pd.DataFrame() if evolution is None else evolution
        self._importance_cputime: Optional[float] = importance_cputime

    @attribute(domain=Dataset.datasets)
    def dataset(self) -> str:
        """Dataset to use for training and validation."""
        return self._dataset

    @attribute(domain=Pipeline.pipelines)
    def pipeline(self) -> str:
        """Pipeline to use for feature extraction."""
        return self._pipeline

    @attribute
    def method(self) -> RepairMethod:
        """Method used to perform data repairs."""
        return self._method

    @attribute(domain=[None])
    def model(self) -> ModelSpec:
        """Model used to make predictions."""
        return self._model

    @attribute
    def trainbias(self) -> float:
        """The bias of the training dataset used in fairness experiments."""
        return self._trainbias

    @attribute
    def valbias(self) -> float:
        """The bias of the validation dataset used in fairness experiments."""
        return self._valbias

    @attribute(domain=[None])
    def biasmethod(self) -> BiasMethod:
        """The method to use when applying a bias to datasets."""
        return self._biasmethod

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
        """The size of the training dataset to use for model training. The value 0 means maximal value."""
        return self._trainsize

    @attribute
    def valsize(self) -> int:
        """The size of the validation dataset to use for importance computation. The value 0 means maximal value."""
        return self._valsize

    @attribute
    def testsize(self) -> int:
        """The size of the test dataset to use for final evaluation. The value 0 means maximal value."""
        return self._testsize

    @attribute
    def mc_timeout(self) -> int:
        """The maximum time in seconds that a Monte-Carlo importance method is allowed to run. Zero means no timeout."""
        return self._mc_timeout

    @attribute
    def mc_tolerance(self) -> float:
        """The parameter that controls Truncated Monte-Carlo early stopping. Zero means no early stopping."""
        return self._mc_tolerance

    @attribute
    def nn_k(self) -> int:
        """The size of the K-neighborhood of the Shapley KNN model."""
        return self._nn_k

    @attribute
    def checkpoints(self) -> int:
        """
        The number of checkpoints to record in the workflow.
        The value 0 means that a checkpoint will be recorded for each training data example.
        """
        return self._checkpoints

    @attribute
    def providers(self) -> int:
        """
        The number of data providers. Each provider is treated as one unit.
        The value 0 means that each data example is provided independently.
        """
        return self._providers

    @attribute(domain=[None])
    def repairgoal(self) -> RepairGoal:
        """The goal of repairing data which impacts the behavior of the scenario."""
        return self._repairgoal

    @result
    def evolution(self) -> DataFrame:
        """The evolution of the experimental parameters."""
        return self._evolution

    @property
    def completed(self) -> bool:
        return len(self._evolution) == self.checkpoints + 1

    @property
    def dataframe(self) -> DataFrame:
        result = self._evolution.assign(
            id=self.id,
            dataset=self.dataset,
            pipeline=self.pipeline,
            model=self.model,
            providers=self.providers,
            mc_timeout=self.mc_timeout,
            mc_tolerance=self.mc_tolerance,
            nn_k=self.nn_k,
            method=self.method,
            utility=self.utility,
            iteration=self.iteration,
        )
        if "importance_cputime" not in result.columns and self._importance_cputime is not None:
            result = result.assign(importance_cputime=self._importance_cputime)
        return result

    @property
    def keyword_replacements(self) -> Dict[str, str]:
        return {**KEYWORD_REPLACEMENTS, **Pipeline.summaries}

    @classmethod
    def is_valid_config(cls, **attributes: Any) -> bool:
        result = True
        if "pipeline" in attributes and "dataset" in attributes:
            dataset = Dataset.datasets[attributes["dataset"]]()
            pipeline = Pipeline.pipelines[attributes["pipeline"]](steps=[("hack", FunctionTransformer())])
            result = result and dataset.modality in pipeline.modalities
        return result and super().is_valid_config(**attributes)
