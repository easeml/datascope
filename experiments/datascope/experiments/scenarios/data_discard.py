from copy import deepcopy
import numpy as np
import pandas as pd

from datascope.importance import (
    Importance,
    JointUtility,
    ShapleyImportance,
    SklearnModelAccuracy,
    SklearnModelEqualizedOddsDifference,
    SklearnModelRocAuc,
    Utility,
)
from datascope.importance.utility import compute_groupings
from datetime import timedelta
from methodtools import lru_cache
from numpy.typing import NDArray
from time import process_time_ns
from typing import Any, Optional, Dict, Type

from .data_repair_scenario import (
    DataRepairScenario,
    get_relative_score,
    RepairMethod,
    IMPORTANCE_METHODS,
    MC_ITERATIONS,
    DEFAULT_SEED,
    DEFAULT_CHECKPOINTS,
    DEFAULT_PROVIDERS,
    DEFAULT_MODEL,
    DEFAULT_REPAIR_GOAL,
    DEFAULT_TRAIN_BIAS,
    DEFAULT_VAL_BIAS,
    DEFAULT_AUGMENT_FACTOR,
    DEFAULT_MC_TIMEOUT,
    DEFAULT_MC_TOLERANCE,
    DEFAULT_NN_K,
    UtilityType,
    RepairGoal,
)
from ..bench import attribute
from ..datasets import (
    Dataset,
    RandomDataset,
    BiasMethod,
    BiasedMixin,
    AugmentableMixin,
    TabularDatasetMixin,
    ImageDatasetMixin,
    DEFAULT_BIAS_METHOD,
    DEFAULT_CACHE_DIR,
)
from ..pipelines import Pipeline, FlattenPipeline, Model, DistanceModelMixin, Postprocessor, LogisticRegressionModel


DEFAULT_MAX_REMOVE = 0.5
KEYWORD_REPLACEMENTS = {
    "discarded": "Number of Data Examples Removed",
    "discarded_rel": "Portion of Data Examples Removed",
    "dataset_bias": "Dataset Bias",
    "mean_label": "Portion of Positive Labels",
}


class DataDiscardScenario(DataRepairScenario, id="data-discard"):
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
        checkpoints: int = DEFAULT_CHECKPOINTS,
        providers: int = DEFAULT_PROVIDERS,
        repairgoal: RepairGoal = DEFAULT_REPAIR_GOAL,
        evolution: Optional[pd.DataFrame] = None,
        importance_cputime: Optional[float] = None,
        iteration: Optional[int] = None,  # Iteration became seed. Keeping this for back compatibility reasons.
        maxremove: float = DEFAULT_MAX_REMOVE,
        **kwargs: Any
    ) -> None:
        super().__init__(
            dataset=dataset,
            pipeline=pipeline,
            method=method,
            utility=utility,
            model=model,
            postprocessor=postprocessor,
            trainbias=trainbias,
            valbias=valbias,
            biasmethod=biasmethod,
            seed=seed,
            augment_factor=augment_factor,
            eager_preprocessing=eager_preprocessing,
            pipeline_cache_dir=pipeline_cache_dir,
            mc_timeout=mc_timeout,
            mc_tolerance=mc_tolerance,
            nn_k=nn_k,
            checkpoints=checkpoints,
            providers=providers,
            repairgoal=repairgoal,
            evolution=evolution,
            importance_cputime=importance_cputime,
            iteration=iteration,
            **kwargs
        )
        self._maxremove = maxremove

    @attribute
    def maxremove(self) -> float:
        """The maximum portion of data to remove."""
        return self._maxremove

    @lru_cache(maxsize=1)
    @classmethod
    def get_keyword_replacements(cls: Type["DataDiscardScenario"]) -> Dict[str, str]:
        result = super().get_keyword_replacements()
        return {**result, **KEYWORD_REPLACEMENTS}

    @classmethod
    def is_valid_config(cls, **attributes: Any) -> bool:
        result = True
        dataset: Optional[Dataset] = attributes.get("dataset", None)
        repairgoal: RepairGoal = attributes.get("repairgoal", None)
        utility: UtilityType = attributes.get("utility", None)
        method: RepairMethod = attributes.get("method", None)
        model: Model = attributes.get("model", None)
        if repairgoal is not None and repairgoal == RepairGoal.FAIRNESS:
            if dataset is not None:
                result = result and isinstance(dataset, BiasedMixin)
        elif repairgoal is not None and repairgoal == RepairGoal.ACCURACY:
            if dataset is not None:
                result = result and not isinstance(dataset, RandomDataset)
            if utility is not None:
                result = result and utility in [UtilityType.ACCURACY, UtilityType.ROC_AUC]
        if method is not None and method == RepairMethod.KNN_Raw:
            result = result and any(
                isinstance(dataset, modality) for modality in [TabularDatasetMixin, ImageDatasetMixin]
            )
        elif method is not None and method == RepairMethod.INFLUENCE:
            result = result and isinstance(model, LogisticRegressionModel)

        return result and super().is_valid_config(**attributes)

    def _run(self, progress_bar: bool = True, **kwargs: Any) -> None:
        # Load dataset.
        if self._trainbias != 0.0 or self._valbias != 0.0:
            assert isinstance(self.dataset, BiasedMixin)
            dataset = self.dataset.load_biased(
                train_bias=self._trainbias, val_bias=self._valbias, bias_method=self._biasmethod
            )
        else:
            dataset = self.dataset.load()
        self.logger.debug("Loaded dataset %s.", dataset)

        if self.augment_factor > 0:
            assert isinstance(dataset, AugmentableMixin)
            dataset.augment(factor=self.augment_factor, inplace=True)

        # Compute sensitive feature groupings.
        groupings_val: Optional[NDArray] = None
        groupings_test: Optional[NDArray] = None
        if isinstance(dataset, BiasedMixin):
            assert isinstance(dataset, Dataset)
            groupings_val = compute_groupings(dataset.X_val, dataset.sensitive_feature)
            groupings_test = compute_groupings(dataset.X_test, dataset.sensitive_feature)

        # Load the pipeline and process the data if eager preprocessing was specified.
        pipeline = self.pipeline.construct(dataset)
        if self.eager_preprocessing:
            dataset.apply(pipeline, cache_dir=self.pipeline_cache_dir, inplace=True)

        # Initialize the model, postprocessor and utility.
        model = self.model.construct(dataset)
        postprocessor = None if self.postprocessor is None else self.postprocessor.construct(dataset)

        target_utility: Utility
        if self.utility == UtilityType.ACCURACY:
            target_utility = JointUtility(SklearnModelAccuracy(model, postprocessor=postprocessor), weights=[-1.0])
        elif self.utility == UtilityType.EQODDS:
            assert self.repairgoal == RepairGoal.FAIRNESS and isinstance(dataset, BiasedMixin)
            target_utility = SklearnModelEqualizedOddsDifference(
                model,
                sensitive_features=dataset.sensitive_feature,
                groupings=groupings_val,
                postprocessor=postprocessor,
            )
        elif self.utility == UtilityType.EQODDS_AND_ACCURACY:
            assert self.repairgoal == RepairGoal.FAIRNESS and isinstance(dataset, BiasedMixin)
            target_utility = JointUtility(
                SklearnModelAccuracy(model, postprocessor=postprocessor),
                SklearnModelEqualizedOddsDifference(
                    model,
                    sensitive_features=dataset.sensitive_feature,
                    groupings=groupings_val,
                    postprocessor=postprocessor,
                ),
                weights=[-0.5, 0.5],
            )
        elif self.utility == UtilityType.ROC_AUC:
            target_utility = JointUtility(SklearnModelRocAuc(model, postprocessor=postprocessor), weights=[-1.0])
        else:
            raise ValueError("Unknown utility type '%s'." % repr(self.utility))
        compute_eqodds = self.repairgoal == RepairGoal.FAIRNESS

        # Compute importance scores and time it.
        random = np.random.RandomState(seed=self.seed)
        importance_time_start = process_time_ns()
        importance: Optional[Importance] = None
        importances: Optional[NDArray] = None
        if self.method == RepairMethod.RANDOM:
            importances = np.array(random.rand(dataset.provenance.num_units))
        elif self.method == RepairMethod.INFLUENCE:
            from ..baselines.influence.importance import InfluenceImportance

            dataset_f = dataset.apply(pipeline)
            importance = InfluenceImportance()
            importances = np.array(
                importance.fit(dataset_f.X_train, dataset_f.y_train, provenance=dataset_f.provenance).score(
                    dataset_f.X_val, dataset_f.y_val
                )
            )
            importances = np.negative(importances)
        else:
            shapley_pipeline = pipeline if self.method != RepairMethod.KNN_Raw else FlattenPipeline()
            method = IMPORTANCE_METHODS[self.method]
            mc_iterations = MC_ITERATIONS[self.method]
            mc_preextract = RepairMethod.is_tmc_nonpipe(self.method)
            importance = ShapleyImportance(
                method=method,
                utility=target_utility,
                pipeline=None if self.eager_preprocessing else shapley_pipeline,
                mc_iterations=mc_iterations,
                mc_timeout=self.mc_timeout,
                mc_tolerance=self.mc_tolerance,
                mc_preextract=mc_preextract,
                nn_k=self.nn_k,
            )
            if isinstance(model, DistanceModelMixin):
                importance.nn_distance = model.distance
            importances = np.array(
                importance.fit(
                    dataset.X_train, dataset.y_train, dataset.metadata_train, provenance=dataset.provenance
                ).score(dataset.X_val, dataset.y_val, dataset.metadata_val)
            )
        importance_time_end = process_time_ns()
        importance_cputime = (importance_time_end - importance_time_start) / 1e9
        self.logger.debug("Importance computed in: %s", str(timedelta(seconds=importance_cputime)))
        n_units = dataset.provenance.num_units
        discarded_units = np.zeros(n_units, dtype=bool)
        argsorted_importances = (-np.array(importances)).argsort()

        # Run the model to get initial score.
        dataset_f = dataset if self.eager_preprocessing else dataset.apply(pipeline, cache_dir=self.pipeline_cache_dir)
        scores = self.compute_model_quality_scores(
            model=model,
            dataset=dataset_f,
            postprocessor=postprocessor,
            compute_eqodds=compute_eqodds,
            groupings_val=groupings_val,
            groupings_test=groupings_test,
        )

        # Update result table.
        dataset_bias = dataset.train_bias if isinstance(dataset, BiasedMixin) else None
        evolution = [
            [
                0.0,
                scores.get("test_eqodds", 0.0),
                scores.get("test_accuracy", 0.0),
                scores.get("test_roc_auc", 0.0),
                scores.get("val_eqodds", 0.0),
                scores.get("val_accuracy", 0.0),
                scores.get("val_roc_auc", 0.0),
                0,
                0.0,
                dataset_bias,
                # dataset.y_train.mean(),
                0.0,
            ]
        ]

        # Set up progress bar.
        n_checkpoints = self.numcheckpoints if self.numcheckpoints > 0 and self.numcheckpoints < n_units else n_units
        rel_units_per_checkpoint = self._maxremove / n_checkpoints
        n_units_per_checkpoint_f = rel_units_per_checkpoint * n_units
        n_units_covered_f = 0.0
        n_units_covered = 0
        if progress_bar:
            self.progress.start(total=n_checkpoints, desc="(id=%s) Repairs" % self.id)

        # Iterate over the repair process.
        dataset_current = deepcopy(dataset)
        discarded_rel = 0.0
        for i in range(n_checkpoints):
            # Update the count of units that were covered and that should be covered in this checkpoint.
            n_units_covered_f += n_units_per_checkpoint_f
            n_units_per_checkpoint = round(n_units_covered_f) - n_units_covered
            n_units_covered += n_units_per_checkpoint

            # Determine indices of data examples that should be discarded given the unit with the highest importance.
            present_units = np.invert(discarded_units)
            target_units = argsorted_importances[present_units[argsorted_importances]]
            target_units = target_units[:n_units_per_checkpoint]
            discarded_units[target_units] = True
            present_units = np.invert(discarded_units).astype(int)
            present_idx = dataset.provenance.query(present_units)
            dataset_current = dataset.select_train(present_idx)  # type: ignore
            discarded_rel += rel_units_per_checkpoint

            # Compute quality metrics.
            dataset_current_f = (
                dataset_current
                if self.eager_preprocessing
                else dataset_current.apply(pipeline, cache_dir=self.pipeline_cache_dir)
            )
            scores = self.compute_model_quality_scores(
                model=model,
                dataset=dataset_current_f,
                postprocessor=postprocessor,
                compute_eqodds=compute_eqodds,
                groupings_val=groupings_val,
                groupings_test=groupings_test,
            )

            # Update result table.
            steps_rel = (i + 1) / float(n_checkpoints)
            discarded = discarded_units.sum(dtype=int)
            # discarded_rel = discarded / float(n_units)
            dataset_bias = dataset_current.train_bias if isinstance(dataset_current, BiasedMixin) else None
            evolution.append(
                [
                    steps_rel,
                    scores.get("test_eqodds", 0.0),
                    scores.get("test_accuracy", 0.0),
                    scores.get("test_roc_auc", 0.0),
                    scores.get("val_eqodds", 0.0),
                    scores.get("val_accuracy", 0.0),
                    scores.get("val_roc_auc", 0.0),
                    discarded,
                    discarded_rel,
                    dataset_bias,
                    # mean_label,
                    importance_cputime,
                ]
            )

            # Recompute if needed.
            if importance is not None and self.method == RepairMethod.KNN_Interactive:
                importance_time_start = process_time_ns()
                assert importances is not None
                importances[present_idx] = importance.fit(
                    dataset_current.X_train,
                    dataset_current.y_train,
                    dataset_current.metadata_train,
                    provenance=dataset_current.provenance,
                ).score(dataset_current.X_val, dataset_current.y_val, dataset_current.metadata_val)
                importance_time_end = process_time_ns()
                importance_cputime += (importance_time_end - importance_time_start) / 1e9
                argsorted_importances = (-np.array(importances)).argsort()

            # Update progress bar.
            if progress_bar:
                self.progress.update(1)

        # Ensure index column has a label.
        self._evolution = pd.DataFrame(
            evolution,
            columns=[
                "steps_rel",
                "test_eqodds",
                "test_accuracy",
                "test_roc_auc",
                "val_eqodds",
                "val_accuracy",
                "val_roc_auc",
                "discarded",
                "discarded_rel",
                "dataset_bias",
                # "mean_label",
                "importance_cputime",
            ],
        )
        if compute_eqodds:
            self._evolution["test_eqodds_rel"] = get_relative_score(
                self._evolution["test_eqodds"], lower_is_better=True
            )
            self._evolution["val_eqodds_rel"] = get_relative_score(self._evolution["val_eqodds"], lower_is_better=True)
        else:
            self._evolution.drop(columns=["test_eqodds", "val_eqodds"], inplace=True)
        self._evolution["test_accuracy_rel"] = get_relative_score(self._evolution["test_accuracy"])
        self._evolution["val_accuracy_rel"] = get_relative_score(self._evolution["val_accuracy"])
        self._evolution["test_roc_auc_rel"] = get_relative_score(self._evolution["test_roc_auc"])
        self._evolution["val_roc_auc_rel"] = get_relative_score(self._evolution["val_roc_auc"])
        self._evolution.index.name = "steps"

        # Close progress bar.
        if progress_bar:
            self.progress.close()
