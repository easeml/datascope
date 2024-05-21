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
from typing import Any, Iterable, List, Optional, Union, Dict, Type

from ..bench import attribute
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
    DEFAULT_TRAIN_BIAS,
    DEFAULT_VAL_BIAS,
    DEFAULT_AUGMENT_FACTOR,
    DEFAULT_REPAIR_GOAL,
    DEFAULT_MC_TIMEOUT,
    DEFAULT_MC_TOLERANCE,
    DEFAULT_NN_K,
    UtilityType,
    RepairGoal,
)
from ..datasets import (
    Dataset,
    BiasMethod,
    BiasedNoisyLabelDataset,
    NoisyLabelDataset,
    AugmentableMixin,
    TabularDatasetMixin,
    ImageDatasetMixin,
    RandomDataset,
    DEFAULT_BIAS_METHOD,
    DEFAULT_CACHE_DIR,
)
from ..pipelines import Pipeline, FlattenPipeline, Model, DistanceModelMixin, Postprocessor, LogisticRegressionModel


DEFAULT_DIRTY_RATIO = 0.5
DEFAULT_DIRTY_BIAS = 0.0
KEYWORD_REPLACEMENTS = {
    "repaired": "Number of Labels Examined",
    "repaired_rel": "Portion of Labels Examined",
    "discovered": "Number of Dirty Labels Found",
    "discovered_rel": "Portion of Dirty Labels Found",
}


class LabelRepairScenario(DataRepairScenario, id="label-repair"):
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
        dirtyratio: float = DEFAULT_DIRTY_RATIO,
        dirtybias: float = DEFAULT_DIRTY_BIAS,
        **kwargs: Any
    ) -> None:
        kwargs.pop("biasmethod", None)
        super().__init__(
            dataset=dataset,
            pipeline=pipeline,
            method=method,
            utility=utility,
            model=model,
            postprocessor=postprocessor,
            trainbias=trainbias,
            valbias=valbias,
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
        self._dirtyratio = dirtyratio
        self._dirtybias = dirtybias

    @classmethod
    def is_valid_config(cls, **attributes: Any) -> bool:
        result = True
        dataset: Optional[Dataset] = attributes.get("dataset", None)
        repairgoal: RepairGoal = attributes.get("repairgoal", None)
        utility: UtilityType = attributes.get("utility", None)
        method: RepairMethod = attributes.get("method", None)
        model: Model = attributes.get("model", None)

        if dataset is not None:
            result = result and isinstance(dataset, NoisyLabelDataset)
        if repairgoal is not None or repairgoal == RepairGoal.FAIRNESS:
            if dataset is not None:
                result = result and isinstance(dataset, BiasedNoisyLabelDataset)
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

    @attribute
    def dirtyratio(self) -> float:
        """The proportion of examples that will have corrupted labels in label repair experiments."""
        return self._dirtyratio

    @attribute
    def dirtybias(self) -> float:
        """The bias of dirty ratio between the sensitive data subgroup and the rest."""
        return self._dirtybias

    @lru_cache(maxsize=1)
    @classmethod
    def get_keyword_replacements(cls: Type["LabelRepairScenario"]) -> Dict[str, str]:
        result = super().get_keyword_replacements()
        return {**result, **KEYWORD_REPLACEMENTS}

    def _run(self, progress_bar: bool = True, **kwargs: Any) -> None:
        # Load dataset.
        dataset = self.dataset.load()
        assert isinstance(dataset, NoisyLabelDataset)
        self.logger.debug("Loaded dataset %s.", dataset)

        # Compute sensitive feature groupings.
        groupings_val: Optional[NDArray] = None
        groupings_test: Optional[NDArray] = None
        if isinstance(dataset, BiasedNoisyLabelDataset):
            groupings_val = compute_groupings(dataset.X_val, dataset.sensitive_feature)
            groupings_test = compute_groupings(dataset.X_test, dataset.sensitive_feature)

        # Create the dirty dataset and apply the data corruption.
        probabilities: Union[float, List[float]] = self._dirtyratio
        if self.providers > 1:
            dirty_providers = round(self.providers * self._dirtyratio)
            clean_providers = self.providers - dirty_providers
            probabilities = [float(i + 1) / dirty_providers for i in range(dirty_providers)]
            probabilities += [0.0 for _ in range(clean_providers)]

        if self.repairgoal == RepairGoal.ACCURACY:
            dataset_dirty = dataset.corrupt_labels(probabilities=probabilities)
            units_dirty = deepcopy(dataset_dirty.units_dirty)
        if self.repairgoal == RepairGoal.FAIRNESS:
            assert isinstance(dataset, BiasedNoisyLabelDataset)
            dataset_dirty = dataset.corrupt_labels_with_bias(probabilities=probabilities, groupbias=self.dirtybias)
            units_dirty = deepcopy(dataset_dirty.units_dirty)
        compute_eqodds = self.repairgoal == RepairGoal.FAIRNESS

        if self.augment_factor > 0 and self.method != RepairMethod.KNN_Raw:
            assert isinstance(dataset, AugmentableMixin) and isinstance(dataset_dirty, AugmentableMixin)
            dataset.augment(factor=self.augment_factor, inplace=True)
            dataset_dirty.augment(factor=self.augment_factor, inplace=True)

        # Load the pipeline and process the data if eager preprocessing was specified.
        pipeline = self.pipeline.construct(dataset)
        if self.eager_preprocessing:
            dataset.apply(pipeline, cache_dir=self.pipeline_cache_dir, inplace=True)
            dataset_dirty.apply(pipeline, cache_dir=self.pipeline_cache_dir, inplace=True)

        # Initialize the model, postprocessor and utility.
        model = self.model.construct(dataset)
        postprocessor = None if self.postprocessor is None else self.postprocessor.construct(dataset)

        target_utility: Utility
        if self.utility == UtilityType.ACCURACY:
            target_utility = JointUtility(SklearnModelAccuracy(model, postprocessor=postprocessor), weights=[-1.0])
        elif self.utility == UtilityType.EQODDS:
            assert self.repairgoal == RepairGoal.FAIRNESS and isinstance(dataset, BiasedNoisyLabelDataset)
            target_utility = SklearnModelEqualizedOddsDifference(
                model,
                sensitive_features=dataset.sensitive_feature,
                groupings=groupings_val,
                postprocessor=postprocessor,
            )
        elif self.utility == UtilityType.EQODDS_AND_ACCURACY:
            assert self.repairgoal == RepairGoal.FAIRNESS and isinstance(dataset, BiasedNoisyLabelDataset)
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

        # Compute importance scores and time it.
        importance_time_start = process_time_ns()
        n_units = dataset.provenance.num_units
        importance: Optional[Importance] = None
        importances: Optional[Iterable[float]] = None
        random = np.random.RandomState(seed=self.seed + 1)
        if self.method == RepairMethod.RANDOM:
            importances = list(random.rand(n_units))
        elif self.method == RepairMethod.INFLUENCE:
            from ..baselines.influence.importance import InfluenceImportance

            dataset_f = dataset.apply(pipeline, cache_dir=self.pipeline_cache_dir)
            dataset_dirty_f = dataset_dirty.apply(pipeline, cache_dir=self.pipeline_cache_dir)
            importance = InfluenceImportance()
            importances = importance.fit(
                dataset_dirty_f.X_train, dataset_dirty_f.y_train, provenance=dataset_dirty_f.provenance
            ).score(dataset_f.X_val, dataset_f.y_val)
            importances = [-x for x in importances]
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
                logger=self.logger,
            )
            if isinstance(model, DistanceModelMixin):
                importance.nn_distance = model.distance
            importances = importance.fit(
                dataset_dirty.X_train,
                dataset_dirty.y_train,
                dataset_dirty.metadata_train,
                provenance=dataset_dirty.provenance,
            ).score(dataset.X_val, dataset.y_val, dataset.metadata_val)
        importance_time_end = process_time_ns()
        importance_cputime = (importance_time_end - importance_time_start) / 1e9
        self.logger.debug("Importance computed in: %s", str(timedelta(seconds=importance_cputime)))
        visited_units = np.zeros(n_units, dtype=bool)
        argsorted_importances = (-np.array(importances)).argsort()

        if self.augment_factor > 0 and self.method == RepairMethod.KNN_Raw:
            assert isinstance(dataset, AugmentableMixin) and isinstance(dataset_dirty, AugmentableMixin)
            dataset.augment(factor=self.augment_factor, inplace=True)
            dataset_dirty.augment(factor=self.augment_factor, inplace=True)

        # Run the model to get initial score.
        dataset_f = dataset if self.eager_preprocessing else dataset.apply(pipeline, cache_dir=self.pipeline_cache_dir)
        dataset_dirty_f = (
            dataset_dirty
            if self.eager_preprocessing
            else dataset_dirty.apply(pipeline, cache_dir=self.pipeline_cache_dir)
        )
        scores = self.compute_model_quality_scores(
            model=model,
            dataset=dataset_dirty_f,
            postprocessor=postprocessor,
            compute_eqodds=compute_eqodds,
            groupings_val=groupings_val,
            groupings_test=groupings_test,
        )

        # Update result table.
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
                0,
                0.0,
                0.0,
            ]
        ]

        # Set up progress bar.
        n_checkpoints = self.numcheckpoints if self.numcheckpoints > 0 and self.numcheckpoints < n_units else n_units
        n_units_per_checkpoint = round(n_units / n_checkpoints)
        if progress_bar:
            self.progress.start(total=n_checkpoints, desc="(id=%s) Repairs" % self.id)

        # Iterate over the repair process.
        for i in range(n_checkpoints):
            # Determine indices of data examples that should be repaired given the unit with the highest importance.
            unvisited_units = np.invert(visited_units)
            target_units = argsorted_importances[unvisited_units[argsorted_importances]]
            if i + 1 < n_checkpoints:
                target_units = target_units[:n_units_per_checkpoint]

            # Repair the data example.
            visited_units[target_units] = True
            dataset_dirty.units_dirty[target_units] = False
            dataset_dirty_f.units_dirty[target_units] = False

            # Compute quality metrics.
            scores = self.compute_model_quality_scores(
                model=model,
                dataset=dataset_dirty_f,
                postprocessor=postprocessor,
                compute_eqodds=compute_eqodds,
                groupings_val=groupings_val,
                groupings_test=groupings_test,
            )

            # Update result table.
            steps_rel = (i + 1) / float(n_checkpoints)
            repaired = visited_units.sum(dtype=int)
            repaired_rel = repaired / float(n_units)
            discovered = np.logical_and(visited_units, units_dirty).sum(dtype=int)
            discovered_rel = discovered / units_dirty.sum(dtype=float)
            evolution.append(
                [
                    steps_rel,
                    scores.get("test_eqodds", 0.0),
                    scores.get("test_accuracy", 0.0),
                    scores.get("test_roc_auc", 0.0),
                    scores.get("val_eqodds", 0.0),
                    scores.get("val_accuracy", 0.0),
                    scores.get("val_roc_auc", 0.0),
                    repaired,
                    repaired_rel,
                    discovered,
                    discovered_rel,
                    importance_cputime,
                ]
            )

            # Recompute if needed.
            if importance is not None and self.method == RepairMethod.KNN_Interactive:
                importance_time_start = process_time_ns()
                importances = importance.fit(
                    dataset_dirty.X_train,
                    dataset_dirty.y_train,
                    dataset_dirty.metadata_train,
                    provenance=dataset_dirty.provenance,
                ).score(dataset.X_val, dataset.y_val, dataset.metadata_val)
                importance_time_end = process_time_ns()
                importance_cputime += (importance_time_end - importance_time_start) / 1e9
                argsorted_importances = (-np.array(importances)).argsort()
                # argsorted_importances = np.ma.array(importances, mask=visited_units).argsort()

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
                "repaired",
                "repaired_rel",
                "discovered",
                "discovered_rel",
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
