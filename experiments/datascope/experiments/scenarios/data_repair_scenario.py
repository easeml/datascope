import pandas as pd

from datascope.importance.common import SklearnModel, ExtendedModelMixin, Postprocessor as PipelinePostprocessor
from datascope.importance.shapley import ImportanceMethod, DEFAULT_MC_TIMEOUT, DEFAULT_MC_TOLERANCE, DEFAULT_NN_K
from datascope.importance.utility import SklearnModelUtility, SklearnModelEqualizedOddsDifference, roc_auc_score
from enum import Enum
from methodtools import lru_cache
from numpy.typing import NDArray
from pandas import DataFrame, Series
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from typing import Any, Optional, Dict, Type

from ..bench import Scenario, attribute, result
from ..datasets import (
    Dataset,
    BiasMethod,
    BiasedMixin,
    AugmentableMixin,
    DEFAULT_BIAS_METHOD,
    DEFAULT_CACHE_DIR,
)
from ..pipelines import Pipeline, Model, Postprocessor, LogisticRegressionModel


class RepairMethod(str, Enum):
    KNN_Single = "shapley-knn-single"
    KNN_Interactive = "shapley-knn-interactive"
    KNN_Raw = "shapley-knn-raw"
    TMC_1 = "shapley-tmc-001"
    TMC_5 = "shapley-tmc-005"
    TMC_10 = "shapley-tmc-010"
    TMC_50 = "shapley-tmc-050"
    TMC_100 = "shapley-tmc-100"
    TMC_500 = "shapley-tmc-500"
    TMC_1000 = "shapley-tmc-1000"
    TMC_PIPE_1 = "shapley-tmc-pipe-001"
    TMC_PIPE_5 = "shapley-tmc-pipe-005"
    TMC_PIPE_10 = "shapley-tmc-pipe-010"
    TMC_PIPE_50 = "shapley-tmc-pipe-050"
    TMC_PIPE_100 = "shapley-tmc-pipe-100"
    TMC_PIPE_500 = "shapley-tmc-pipe-500"
    TMC_PIPE_1000 = "shapley-tmc-pipe-1000"
    RANDOM = "random"
    INFLUENCE = "influence"

    @staticmethod
    def is_pipe(method: "RepairMethod") -> bool:
        return method in [
            RepairMethod.TMC_PIPE_1,
            RepairMethod.TMC_PIPE_5,
            RepairMethod.TMC_PIPE_10,
            RepairMethod.TMC_PIPE_50,
            RepairMethod.TMC_PIPE_100,
            RepairMethod.TMC_PIPE_500,
            RepairMethod.TMC_PIPE_1000,
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
            RepairMethod.TMC_1000,
            RepairMethod.TMC_PIPE_1,
            RepairMethod.TMC_PIPE_5,
            RepairMethod.TMC_PIPE_10,
            RepairMethod.TMC_PIPE_50,
            RepairMethod.TMC_PIPE_100,
            RepairMethod.TMC_PIPE_500,
            RepairMethod.TMC_PIPE_1000,
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
            RepairMethod.TMC_1000,
        ]


class RepairGoal(str, Enum):
    FAIRNESS = "fairness"
    ACCURACY = "accuracy"


class UtilityType(str, Enum):
    ACCURACY = "acc"
    EQODDS = "eqodds"
    EQODDS_AND_ACCURACY = "eqodds-acc"
    ROC_AUC = "rocauc"


IMPORTANCE_METHODS = {
    RepairMethod.KNN_Single: ImportanceMethod.NEIGHBOR,
    RepairMethod.KNN_Interactive: ImportanceMethod.NEIGHBOR,
    RepairMethod.KNN_Raw: ImportanceMethod.NEIGHBOR,
    RepairMethod.TMC_1: ImportanceMethod.MONTECARLO,
    RepairMethod.TMC_5: ImportanceMethod.MONTECARLO,
    RepairMethod.TMC_10: ImportanceMethod.MONTECARLO,
    RepairMethod.TMC_50: ImportanceMethod.MONTECARLO,
    RepairMethod.TMC_100: ImportanceMethod.MONTECARLO,
    RepairMethod.TMC_500: ImportanceMethod.MONTECARLO,
    RepairMethod.TMC_1000: ImportanceMethod.MONTECARLO,
    RepairMethod.TMC_PIPE_1: ImportanceMethod.MONTECARLO,
    RepairMethod.TMC_PIPE_5: ImportanceMethod.MONTECARLO,
    RepairMethod.TMC_PIPE_10: ImportanceMethod.MONTECARLO,
    RepairMethod.TMC_PIPE_50: ImportanceMethod.MONTECARLO,
    RepairMethod.TMC_PIPE_100: ImportanceMethod.MONTECARLO,
    RepairMethod.TMC_PIPE_500: ImportanceMethod.MONTECARLO,
    RepairMethod.TMC_PIPE_1000: ImportanceMethod.MONTECARLO,
}

MC_ITERATIONS = {
    RepairMethod.KNN_Single: 0,
    RepairMethod.KNN_Interactive: 0,
    RepairMethod.KNN_Raw: 0,
    RepairMethod.TMC_1: 1,
    RepairMethod.TMC_5: 5,
    RepairMethod.TMC_10: 10,
    RepairMethod.TMC_50: 50,
    RepairMethod.TMC_100: 100,
    RepairMethod.TMC_500: 500,
    RepairMethod.TMC_1000: 1000,
    RepairMethod.TMC_PIPE_1: 1,
    RepairMethod.TMC_PIPE_5: 5,
    RepairMethod.TMC_PIPE_10: 10,
    RepairMethod.TMC_PIPE_50: 50,
    RepairMethod.TMC_PIPE_100: 100,
    RepairMethod.TMC_PIPE_500: 500,
    RepairMethod.TMC_PIPE_1000: 1000,
}

KEYWORD_REPLACEMENTS = {
    "random": "Random",
    "influence": "Influence Function",
    "shapley-tmc": "Shapley TMC",
    "shapley-knn-single": "Shapley KNN Single",
    "shapley-knn-interactive": "Shapley KNN Interactive",
    "shapley-knn-raw": "Shapley KNN over Raw Features",
    "shapley-tmc-001": "Shapley TMC+PP x1",
    "shapley-tmc-005": "Shapley TMC+PP x5",
    "shapley-tmc-010": "Shapley TMC+PP x10",
    "shapley-tmc-050": "Shapley TMC+PP x50",
    "shapley-tmc-100": "Shapley TMC+PP x100",
    "shapley-tmc-500": "Shapley TMC+PP x500",
    "shapley-tmc-1000": "Shapley TMC+PP x1000",
    "shapley-tmc-pipe-001": "Shapley TMC x1",
    "shapley-tmc-pipe-005": "Shapley TMC x5",
    "shapley-tmc-pipe-010": "Shapley TMC x10",
    "shapley-tmc-pipe-050": "Shapley TMC x50",
    "shapley-tmc-pipe-100": "Shapley TMC x100",
    "shapley-tmc-pipe-500": "Shapley TMC x500",
    "shapley-tmc-pipe-1000": "Shapley TMC x1000",
    "eqodds": "Equalized Odds Difference",
    "importance_cputime": "Compute Time [s]",
    "steps": "Repair Steps Taken",
    "steps_rel": "Relative Repair Steps Taken",
    "acc": "Accuracy",
    "eqodds-acc": "Accuracy + Equalized Odds Difference",
    "accuracy": "Accuracy",
    "accuracy_rel": "Relative Accuracy",
    "roc_auc": "ROC AUC",
    "roc_auc_rel": "Relative ROC AUC",
    "train_eqodds": "Equalized Odds Difference",
    "train_eqodds_rel": "Relative Equalized Odds Difference",
    "train_accuracy": "Accuracy",
    "train_accuracy_rel": "Relative Accuracy",
    "train_roc_auc": "ROC AUC",
    "train_roc_auc_rel": "Relative ROC AUC",
    "val_eqodds": "Validation Equalized Odds Difference",
    "val_eqodds_rel": "Validation Relative Equalized Odds Difference",
    "val_accuracy": "Validation Accuracy",
    "val_accuracy_rel": "Validation Relative Accuracy",
    "val_roc_auc": "Validation ROC AUC",
    "val_roc_auc_rel": "Validation Relative ROC AUC",
    "test_eqodds": "Test Equalized Odds Difference",
    "test_eqodds_rel": "Test Relative Equalized Odds Difference",
    "test_accuracy": "Test Accuracy",
    "test_accuracy_rel": "Test Relative Accuracy",
    "test_roc_auc": "Test ROC AUC",
    "test_roc_auc_rel": "Test Relative ROC AUC",
}

DEFAULT_SEED = 1
DEFAULT_CHECKPOINTS = 100
DEFAULT_PROVIDERS = 0
DEFAULT_MODEL = LogisticRegressionModel()
DEFAULT_REPAIR_GOAL = RepairGoal.ACCURACY
DEFAULT_TRAIN_BIAS = 0.0
DEFAULT_VAL_BIAS = 0.0
DEFAULT_AUGMENT_FACTOR = 0


def get_relative_score(scores: Series, lower_is_better: bool = False) -> Series:
    start_score = scores.iloc[0]
    if lower_is_better:
        min_score = min(scores)
        delta_score = abs(start_score - min_score)
        return scores.apply(lambda x: (x - min_score) / (delta_score + 1e-9))
    else:
        max_score = max(scores)
        delta_score = abs(start_score - max_score)
        return scores.apply(lambda x: (x - start_score) / (delta_score + 1e-9))


class DataRepairScenario(Scenario, abstract=True):
    def __init__(
        self,
        dataset: Dataset,
        pipeline: Pipeline,
        method: RepairMethod,
        utility: UtilityType,
        model: Model = DEFAULT_MODEL,
        postprocessor: Optional[Postprocessor] = None,
        trainbias: float = DEFAULT_TRAIN_BIAS,
        valbias: float = DEFAULT_VAL_BIAS,
        biasmethod: BiasMethod = DEFAULT_BIAS_METHOD,
        seed: int = DEFAULT_SEED,
        augment_factor: int = DEFAULT_AUGMENT_FACTOR,
        eager_preprocessing: bool = False,
        pipeline_cache_dir: str = DEFAULT_CACHE_DIR,
        mc_timeout: int = DEFAULT_MC_TIMEOUT,
        mc_tolerance: float = DEFAULT_MC_TOLERANCE,
        nn_k: int = DEFAULT_NN_K,
        numcheckpoints: int = DEFAULT_CHECKPOINTS,
        providers: int = DEFAULT_PROVIDERS,
        repairgoal: RepairGoal = DEFAULT_REPAIR_GOAL,
        evolution: Optional[pd.DataFrame] = None,
        importance_cputime: Optional[float] = None,
        iteration: Optional[int] = None,  # Iteration became seed. Keeping this for back compatibility reasons.
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self._dataset = dataset
        self._pipeline = pipeline
        self._method = method
        self._utility = utility
        self._model = model
        self._postprocessor = postprocessor
        self._trainbias = trainbias
        self._valbias = valbias
        self._biasmethod = biasmethod
        self._seed = seed
        if iteration is not None:
            self._seed += iteration
        self._augment_factor = augment_factor
        self._eager_preprocessing = eager_preprocessing
        self._pipeline_cache_dir = pipeline_cache_dir
        self._mc_timeout = mc_timeout
        self._mc_tolerance = mc_tolerance
        self._nn_k = nn_k
        self._numcheckpoints = numcheckpoints
        self._providers = providers
        self._repairgoal = repairgoal
        self._evolution = pd.DataFrame() if evolution is None else evolution
        self._importance_cputime: Optional[float] = importance_cputime

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

    @attribute
    def postprocessor(self) -> Optional[Postprocessor]:
        """Postprocessor used to perform final processing of model predictions."""
        return self._postprocessor

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
    def seed(self) -> int:
        """The random seed that applies to all pseudo-random processes in the scenario."""
        return self._seed

    @attribute
    def augment_factor(self) -> int:
        """The augmentation factor to apply to the dataset after loading it (if applicable)."""
        return self._augment_factor

    @attribute(domain=[None])
    def eager_preprocessing(self) -> bool:
        """Training data is passed through the preprocessing pipeline before being passed to importance computation."""
        return self._eager_preprocessing

    @attribute
    def pipeline_cache_dir(self) -> str:
        """The directory where the pipeline cache is stored."""
        return self._pipeline_cache_dir

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
    def numcheckpoints(self) -> int:
        """
        The number of checkpoints to record in the workflow.
        The value 0 means that a checkpoint will be recorded for each training data example.
        """
        return self._numcheckpoints

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
        return len(self._evolution) == self.numcheckpoints + 1

    @property
    def dataframe(self) -> DataFrame:
        attributes = self._flatten_attributes(self.attributes)
        result = self._evolution.assign(**attributes)
        if "importance_cputime" not in result.columns and self._importance_cputime is not None:
            result = result.assign(importance_cputime=self._importance_cputime)
        return result

    @lru_cache(maxsize=1)
    @classmethod
    def get_keyword_replacements(cls: Type["Scenario"]) -> Dict[str, str]:
        return {
            **KEYWORD_REPLACEMENTS,
            **Dataset.get_keyword_replacements(),
            **Pipeline.get_keyword_replacements(),
            **Postprocessor.get_keyword_replacements(),
            **Model.get_keyword_replacements(),
        }

    @classmethod
    def is_valid_config(cls, **attributes: Any) -> bool:
        result = True
        dataset: Optional[Dataset] = attributes.get("dataset", None)
        pipeline: Optional[Pipeline] = attributes.get("pipeline", None)
        augment_factor: Optional[int] = attributes.get("augment_factor", None)
        if pipeline is not None and dataset is not None:
            result = result and any(isinstance(dataset, modality) for modality in pipeline.modalities)
        if augment_factor is not None and augment_factor > 0 and dataset is not None:
            result = result and isinstance(dataset, AugmentableMixin)
        return result and super().is_valid_config(**attributes)

    def compute_model_quality_scores(
        self,
        model: SklearnModel,
        dataset: Dataset,
        postprocessor: Optional[PipelinePostprocessor] = None,
        compute_eqodds: bool = False,
        groupings_val: Optional[NDArray] = None,
        groupings_test: Optional[NDArray] = None,
    ) -> Dict[str, Any]:
        result = {}

        # Fit the model to the training data.
        model = clone(model)
        if isinstance(model, ExtendedModelMixin):
            model.fit_extended(
                X=dataset.X_train,
                y=dataset.y_train,
                metadata=dataset.metadata_train,
                X_val=dataset.X_val,
                y_val=dataset.y_val,
                metadata_val=dataset.metadata_val,
            )
        else:
            model.fit(dataset.X_train, dataset.y_train)

        # Initialize the utility and compute the scores.
        utility = SklearnModelUtility(
            model=model,
            metric=accuracy_score,
            auxiliary_metrics={"roc_auc": roc_auc_score},
            auxiliary_metric_requires_probabilities={"roc_auc": True},
            postprocessor=postprocessor,
            model_pretrained=True,
            metric_requires_probabilities=False,
            compute_train_score=True,
        )
        utility_result = utility(
            X_train=dataset.X_val,
            y_train=dataset.y_val,
            X_test=dataset.X_test,
            y_test=dataset.y_test,
            metadata_train=dataset.metadata_val,
            metadata_test=dataset.metadata_test,
            seed=self.seed,
        )
        assert utility_result.auxiliary_train_scores is not None
        result["val_accuracy"] = utility_result.train_score
        result["test_accuracy"] = utility_result.score
        result["val_roc_auc"] = utility_result.auxiliary_train_scores["roc_auc"]
        result["test_roc_auc"] = utility_result.auxiliary_scores["roc_auc"]

        if compute_eqodds:
            assert isinstance(dataset, BiasedMixin)
            eqodds_utility_val = SklearnModelEqualizedOddsDifference(
                model,
                sensitive_features=dataset.sensitive_feature,
                groupings=groupings_val,
                postprocessor=postprocessor,
                model_pretrained=True,
            )
            eqodds_utility_test = SklearnModelEqualizedOddsDifference(
                model,
                sensitive_features=dataset.sensitive_feature,
                groupings=groupings_test,
                postprocessor=postprocessor,
                model_pretrained=True,
            )
            eqodds_result_val = eqodds_utility_val(
                X_train=dataset.X_train,
                y_train=dataset.y_train,
                X_test=dataset.X_val,
                y_test=dataset.y_val,
                metadata_train=dataset.metadata_train,
                metadata_test=dataset.metadata_val,
                seed=self.seed,
            )
            eqodds_result_test = eqodds_utility_test(
                X_train=dataset.X_train,
                y_train=dataset.y_train,
                X_test=dataset.X_test,
                y_test=dataset.y_test,
                metadata_train=dataset.metadata_train,
                metadata_test=dataset.metadata_test,
                seed=self.seed,
            )
            result["val_eqodds"] = eqodds_result_val.score
            result["test_eqodds"] = eqodds_result_test.score

        return result
