from copy import deepcopy
import numpy as np
import pandas as pd

from datascope.importance.common import (
    JointUtility,
    SklearnModelAccuracy,
    SklearnModelEqualizedOddsDifference,
    SklearnModelRocAuc,
    Utility,
    UtilityResult,
    compute_groupings,
)
from datascope.importance.importance import Importance
from datascope.importance.shapley import ShapleyImportance
from datetime import timedelta
from numpy.typing import NDArray
from time import process_time_ns
from typing import Any, Optional, Dict

from .datascope_scenario import (
    DatascopeScenario,
    RepairMethod,
    ModelSpec,
    MODEL_TYPES,
    MODEL_KWARGS,
    IMPORTANCE_METHODS,
    MC_ITERATIONS,
    DEFAULT_SEED,
    DEFAULT_CHECKPOINTS,
    DEFAULT_MODEL,
    DEFAULT_REPAIR_GOAL,
    DEFAULT_TRAIN_BIAS,
    DEFAULT_VAL_BIAS,
    DEFAULT_MC_TIMEOUT,
    DEFAULT_MC_TOLERANCE,
    DEFAULT_NN_K,
    UtilityType,
    RepairGoal,
)
from .base import attribute
from ..datasets import (
    Dataset,
    BiasMethod,
    BiasedMixin,
    AugmentableMixin,
    TabularDatasetMixin,
    ImageDatasetMixin,
    DEFAULT_TRAINSIZE,
    DEFAULT_VALSIZE,
    DEFAULT_TESTSIZE,
    DEFAULT_BIAS_METHOD,
    DEFAULT_CACHE_DIR,
)
from ..pipelines import Pipeline, FlattenPipeline, get_model, DistanceModelMixin


DEFAULT_MAX_REMOVE = 0.5
KEYWORD_REPLACEMENTS = {
    "eqodds": "Equalized Odds Difference",
    "eqodds_rel": "Relative Equalized Odds Difference",
    "accuracy": "Accuracy",
    "accuracy_rel": "Relative Accuracy",
    "discarded": "Number of Data Examples Removed",
    "discarded_rel": "Portion of Data Examples Removed",
    "dataset_bias": "Dataset Bias",
    "mean_label": "Portion of Positive Labels",
}


class DataDiscardScenario(DatascopeScenario, id="data-discard"):
    def __init__(
        self,
        dataset: str,
        pipeline: str,
        method: RepairMethod,
        utility: UtilityType,
        iteration: int,
        model: ModelSpec = DEFAULT_MODEL,
        postprocessor: Optional[str] = None,
        trainbias: float = DEFAULT_TRAIN_BIAS,
        valbias: float = DEFAULT_VAL_BIAS,
        biasmethod: BiasMethod = DEFAULT_BIAS_METHOD,
        maxremove: float = DEFAULT_MAX_REMOVE,
        repairgoal: RepairGoal = DEFAULT_REPAIR_GOAL,
        seed: int = DEFAULT_SEED,
        trainsize: int = DEFAULT_TRAINSIZE,
        valsize: int = DEFAULT_VALSIZE,
        testsize: int = DEFAULT_TESTSIZE,
        mc_timeout: int = DEFAULT_MC_TIMEOUT,
        mc_tolerance: float = DEFAULT_MC_TOLERANCE,
        nn_k: int = DEFAULT_NN_K,
        checkpoints: int = DEFAULT_CHECKPOINTS,
        pipeline_cache_dir: str = DEFAULT_CACHE_DIR,
        evolution: Optional[pd.DataFrame] = None,
        importance_cputime: Optional[float] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(
            dataset=dataset,
            pipeline=pipeline,
            method=method,
            utility=utility,
            iteration=iteration,
            model=model,
            postprocessor=postprocessor,
            trainbias=trainbias,
            valbias=valbias,
            biasmethod=biasmethod,
            seed=seed,
            trainsize=trainsize,
            valsize=valsize,
            testsize=testsize,
            mc_timeout=mc_timeout,
            mc_tolerance=mc_tolerance,
            nn_k=nn_k,
            checkpoints=checkpoints,
            repairgoal=repairgoal,
            pipeline_cache_dir=pipeline_cache_dir,
            evolution=evolution,
            importance_cputime=importance_cputime,
            **kwargs
        )
        self._maxremove = maxremove

    @attribute
    def maxremove(self) -> float:
        """The maximum portion of data to remove."""
        return self._maxremove

    @property
    def keyword_replacements(self) -> Dict[str, str]:
        result = super().keyword_replacements
        return {**result, **KEYWORD_REPLACEMENTS}

    @classmethod
    def is_valid_config(cls, **attributes: Any) -> bool:
        result = True
        if "repairgoal" not in attributes or attributes["repairgoal"] == RepairGoal.FAIRNESS:
            if "dataset" in attributes:
                dataset = Dataset.datasets[attributes["dataset"]]()
                result = result and isinstance(dataset, BiasedMixin)
        elif "repairgoal" in attributes and attributes["repairgoal"] == RepairGoal.ACCURACY:
            if "dataset" in attributes:
                result = result and (attributes["dataset"] != "random")
            if "utility" in attributes:
                result = result and attributes["utility"] in [UtilityType.ACCURACY, UtilityType.ROC_AUC]
        if "method" in attributes and attributes["method"] == RepairMethod.KNN_Raw:
            dataset_class = Dataset.datasets[attributes["dataset"]]
            result = result and any(
                issubclass(dataset_class, modality) for modality in [TabularDatasetMixin, ImageDatasetMixin]
            )
        elif "method" in attributes and attributes["method"] == RepairMethod.INFLUENCE:
            result = result and attributes.get("model", DEFAULT_MODEL) == ModelSpec.LogisticRegression

        return result and super().is_valid_config(**attributes)

    def _run(self, progress_bar: bool = True, **kwargs: Any) -> None:
        # Load dataset.
        seed = self._seed + self._iteration
        dataset = Dataset.datasets[self.dataset](
            trainsize=self.trainsize, valsize=self.valsize, testsize=self.testsize, seed=seed
        )
        if self._trainbias != 0.0 or self._valbias != 0.0:
            assert isinstance(dataset, BiasedMixin)
            dataset.load_biased(train_bias=self._trainbias, val_bias=self._valbias, bias_method=self._biasmethod)
        else:
            dataset.load()
        self.logger.debug(
            "Dataset '%s' loaded (trainsize=%d, valsize=%d, testsize=%d).",
            self.dataset,
            dataset.trainsize,
            dataset.valsize,
            dataset.testsize,
        )

        if self.augment_factor > 0:
            assert isinstance(dataset, AugmentableMixin)
            dataset.augment(factor=self.augment_factor, inplace=True)

        # Compute sensitive feature groupings.
        groupings_val: Optional[NDArray] = None
        groupings_test: Optional[NDArray] = None
        if isinstance(dataset, BiasedMixin):
            groupings_val = compute_groupings(dataset.X_val, dataset.sensitive_feature)
            groupings_test = compute_groupings(dataset.X_test, dataset.sensitive_feature)

        # Load the pipeline and process the data.
        pipeline = Pipeline.pipelines[self.pipeline].construct(dataset)
        # X_train, y_train = dataset.X_train, dataset.y_train
        # X_val, y_val = dataset.X_val, dataset.y_val
        # X_train_orig, y_train_orig = dataset.X_train, dataset.y_train
        # if RepairMethod.is_tmc_nonpipe(self.method):
        #     raise ValueError("This is not supported at the moment.")
        # self.logger.debug("Shape of X_train before feature extraction: %s", str(dataset.X_train.shape))
        # dataset = dataset.apply(pipeline)  # TODO: Fit the pipeline with dirty data.
        # assert isinstance(dataset, BiasedMixin)
        # self.logger.debug("Shape of X_train after feature extraction: %s", str(dataset.X_train.shape))

        # Reshape datasets if needed.
        # if dataset.X_train.ndim > 2:
        #     dataset.X_train[:] = dataset.X_train.reshape(dataset.X_train.shape[0], -1)
        #     dataset.X_val[:] = dataset.X_val.reshape(dataset.X_val.shape[0], -1)
        #     self.logger.debug("Need to reshape. New shape: %s", str(dataset.X_train.shape))

        # # Construct binarized provenance matrix.
        # provenance = np.expand_dims(np.arange(dataset.trainsize, dtype=int), axis=(1, 2, 3))
        # provenance = np.pad(provenance, pad_width=((0, 0), (0, 0), (0, 0), (0, 1)))
        # provenance = binarize(provenance)

        if self.eager_preprocessing:
            dataset.apply(pipeline, cache_dir=self.pipeline_cache_dir, inplace=True)

        # Initialize the model and utility.
        model_type = MODEL_TYPES[self.model]
        model_kwargs = MODEL_KWARGS[self.model]
        model = get_model(model_type, **model_kwargs)
        # model_pipeline = deepcopy(pipeline)
        # pipeline.steps.append(("model", model))
        # if RepairMethod.is_pipe(self.method):
        #     model = model_pipeline
        accuracy_utility = SklearnModelAccuracy(model, postprocessor=self.postprocessor)
        roc_auc_utility = SklearnModelRocAuc(model, postprocessor=self.postprocessor)
        eqodds_utility: Optional[SklearnModelEqualizedOddsDifference] = None
        if self.repairgoal == RepairGoal.FAIRNESS:
            assert isinstance(dataset, BiasedMixin)
            eqodds_utility = SklearnModelEqualizedOddsDifference(
                model,
                sensitive_features=dataset.sensitive_feature,
                groupings=groupings_test,
                postprocessor=self.postprocessor,
            )

        # target_model = model if RepairMethod.is_tmc_nonpipe(self.method) else pipeline
        target_utility: Utility
        if self.utility == UtilityType.ACCURACY:
            target_utility = JointUtility(SklearnModelAccuracy(model, postprocessor=self.postprocessor), weights=[-1.0])
            # target_utility = SklearnModelAccuracy(model)
        elif self.utility == UtilityType.EQODDS:
            assert self.repairgoal == RepairGoal.FAIRNESS and isinstance(dataset, BiasedMixin)
            target_utility = SklearnModelEqualizedOddsDifference(
                model,
                sensitive_features=dataset.sensitive_feature,
                groupings=groupings_val,
                postprocessor=self.postprocessor,
            )
        elif self.utility == UtilityType.EQODDS_AND_ACCURACY:
            assert self.repairgoal == RepairGoal.FAIRNESS and isinstance(dataset, BiasedMixin)
            target_utility = JointUtility(
                SklearnModelAccuracy(model, postprocessor=self.postprocessor),
                SklearnModelEqualizedOddsDifference(
                    model,
                    sensitive_features=dataset.sensitive_feature,
                    groupings=groupings_val,
                    postprocessor=self.postprocessor,
                ),
                weights=[-0.5, 0.5],
            )
        elif self.utility == UtilityType.ROC_AUC:
            target_utility = JointUtility(SklearnModelRocAuc(model, postprocessor=self.postprocessor), weights=[-1.0])
        else:
            raise ValueError("Unknown utility type '%s'." % repr(self.utility))

        # Compute importance scores and time it.
        random = np.random.RandomState(seed=seed)
        importance_time_start = process_time_ns()
        importance: Optional[Importance] = None
        importances: Optional[NDArray] = None
        if self.method == RepairMethod.RANDOM:
            importances = np.array(random.rand(dataset.trainsize))
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
        n_units = dataset.trainsize
        discarded_units = np.zeros(n_units, dtype=bool)
        argsorted_importances = (-np.array(importances)).argsort()
        # argsorted_importances = np.ma.array(importances, mask=discarded_units).argsort()

        # Run the model to get initial score.
        dataset_f = dataset if self.eager_preprocessing else dataset.apply(pipeline, cache_dir=self.pipeline_cache_dir)
        val_eqodds_utility_result = UtilityResult()
        test_eqodds_utility_result = UtilityResult()
        if eqodds_utility is not None:
            val_eqodds_utility_result = eqodds_utility(
                dataset_f.X_train, dataset_f.y_train, dataset_f.X_val, dataset_f.y_val
            )
            test_eqodds_utility_result = eqodds_utility(
                dataset_f.X_train, dataset_f.y_train, dataset_f.X_test, dataset_f.y_test
            )
        val_accuracy_utility_result = accuracy_utility(
            dataset_f.X_train, dataset_f.y_train, dataset_f.X_val, dataset_f.y_val
        )
        test_accuracy_utility_result = accuracy_utility(
            dataset_f.X_train, dataset_f.y_train, dataset_f.X_test, dataset_f.y_test
        )
        val_roc_auc_utility_result = roc_auc_utility(
            dataset_f.X_train, dataset_f.y_train, dataset_f.X_val, dataset_f.y_val
        )
        test_roc_auc_utility_result = roc_auc_utility(
            dataset_f.X_train, dataset_f.y_train, dataset_f.X_test, dataset_f.y_test
        )

        # Update result table.
        dataset_bias = dataset.train_bias if isinstance(dataset, BiasedMixin) else None
        evolution = [
            [
                0.0,
                test_eqodds_utility_result.score,
                test_eqodds_utility_result.score,
                test_accuracy_utility_result.score,
                test_accuracy_utility_result.score,
                test_roc_auc_utility_result.score,
                test_roc_auc_utility_result.score,
                val_eqodds_utility_result.score,
                val_eqodds_utility_result.score,
                val_accuracy_utility_result.score,
                val_accuracy_utility_result.score,
                val_roc_auc_utility_result.score,
                val_roc_auc_utility_result.score,
                0,
                0.0,
                dataset_bias,
                dataset.y_train.mean(),
                0.0,
            ]
        ]
        test_eqodds_start = test_eqodds_utility_result.score
        test_accuracy_start = test_accuracy_utility_result.score
        test_roc_auc_start = test_roc_auc_utility_result.score
        val_eqodds_start = val_eqodds_utility_result.score
        val_accuracy_start = val_accuracy_utility_result.score
        val_roc_auc_start = val_roc_auc_utility_result.score

        # Set up progress bar.
        checkpoints = self.checkpoints if self.checkpoints > 0 and self.checkpoints < n_units else n_units
        n_units_per_checkpoint_f = self._maxremove * n_units / checkpoints
        n_units_covered_f = 0.0
        n_units_covered = 0
        if progress_bar:
            self.progress.start(total=checkpoints, desc="(id=%s) Repairs" % self.id)
        # pbar = None if not progress_bar else tqdm(total=dataset.trainsize, desc="%s Repairs" % str(self))

        # Iterate over the repair process.
        # discarded_units = np.zeros(dataset.trainsize, dtype=bool)
        dataset_current = deepcopy(dataset)
        for i in range(checkpoints):
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
            dataset_current = dataset.select_train(present_idx)

            # Display message about current target units that are going to be discarded.
            # y_train_target = y_train_orig[target_units]
            # X_train_target_sf = X_train_orig[target_units, dataset.sensitive_feature]
            # self.logger.debug(
            #     "Discarding %d units. Label 1 ratio: %.2f. Sensitive feature 1 ratio %.2f. Matching ratio: %.2f.",
            #     len(target_units),
            #     y_train_target.mean(),
            #     X_train_target_sf.mean(),
            #     np.mean(y_train_target == X_train_target_sf),
            # )

            # target_query = np.zeros(n_units, dtype=int)
            # target_query[target_units] = 1
            # target_unit = np.ma.array(importances, mask=discarded_units).argmin()
            # target_query = np.eye(1, discarded_units.shape[0], target_unit, dtype=int).flatten()

            # Repair the data example.
            # y_train_dirty[target_idx] = y_train[target_idx]
            # discarded_units[target_units] = True

            # Run the model.
            dataset_current_f = dataset_current.apply(pipeline, cache_dir=self.pipeline_cache_dir)
            if eqodds_utility is not None:
                val_eqodds_utility_result = eqodds_utility(
                    dataset_current_f.X_train,
                    dataset_current_f.y_train,
                    dataset_current_f.X_val,
                    dataset_current_f.y_val,
                    metadata_train=dataset_current_f.metadata_train,
                    metadata_test=dataset_current_f.metadata_test,
                )
                test_eqodds_utility_result = eqodds_utility(
                    dataset_current_f.X_train,
                    dataset_current_f.y_train,
                    dataset_current_f.X_test,
                    dataset_current_f.y_test,
                    metadata_train=dataset_current_f.metadata_train,
                    metadata_test=dataset_current_f.metadata_test,
                )
            val_accuracy_utility_result = accuracy_utility(
                dataset_current_f.X_train,
                dataset_current_f.y_train,
                dataset_current_f.X_val,
                dataset_current_f.y_val,
                metadata_train=dataset_current_f.metadata_train,
                metadata_test=dataset_current_f.metadata_test,
            )
            test_accuracy_utility_result = accuracy_utility(
                dataset_current_f.X_train,
                dataset_current_f.y_train,
                dataset_current_f.X_test,
                dataset_current_f.y_test,
                metadata_train=dataset_current_f.metadata_train,
                metadata_test=dataset_current_f.metadata_test,
            )
            val_roc_auc_utility_result = roc_auc_utility(
                dataset_current_f.X_train,
                dataset_current_f.y_train,
                dataset_current_f.X_val,
                dataset_current_f.y_val,
                metadata_train=dataset_current_f.metadata_train,
                metadata_test=dataset_current_f.metadata_test,
            )
            test_roc_auc_utility_result = roc_auc_utility(
                dataset_current_f.X_train,
                dataset_current_f.y_train,
                dataset_current_f.X_test,
                dataset_current_f.y_test,
                metadata_train=dataset_current_f.metadata_train,
                metadata_test=dataset_current_f.metadata_test,
            )

            # Update result table.
            steps_rel = (i + 1) / float(checkpoints)
            discarded = discarded_units.sum(dtype=int)
            discarded_rel = discarded / float(n_units)
            dataset_bias = dataset_current.train_bias if isinstance(dataset_current, BiasedMixin) else None
            mean_label = np.mean(dataset_current.y_train)
            evolution.append(
                [
                    steps_rel,
                    test_eqodds_utility_result.score,
                    test_eqodds_utility_result.score,
                    test_accuracy_utility_result.score,
                    test_accuracy_utility_result.score,
                    test_roc_auc_utility_result.score,
                    test_roc_auc_utility_result.score,
                    val_eqodds_utility_result.score,
                    val_eqodds_utility_result.score,
                    val_accuracy_utility_result.score,
                    val_accuracy_utility_result.score,
                    val_roc_auc_utility_result.score,
                    val_roc_auc_utility_result.score,
                    discarded,
                    discarded_rel,
                    dataset_bias,
                    mean_label,
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
                "test_eqodds_rel",
                "test_accuracy",
                "test_accuracy_rel",
                "test_roc_auc",
                "test_roc_auc_rel",
                "val_eqodds",
                "val_eqodds_rel",
                "val_accuracy",
                "val_accuracy_rel",
                "val_roc_auc",
                "val_roc_auc_rel",
                "discarded",
                "discarded_rel",
                "dataset_bias",
                "mean_label",
                "importance_cputime",
            ],
        )
        self._evolution.index.name = "steps"

        # Recompute relative equalized odds (if we were keeping track of it).
        if eqodds_utility is not None:
            val_eqodds_end = val_eqodds_utility_result.score
            val_eqqods_min = min(val_eqodds_start, val_eqodds_end)
            val_eqodds_delta = abs(val_eqodds_end - val_eqodds_start)
            self._evolution["val_eqodds_rel"] = self._evolution["val_eqodds_rel"].apply(
                lambda x: (x - val_eqqods_min) / val_eqodds_delta
            )
            test_eqodds_end = test_eqodds_utility_result.score
            test_eqqods_min = min(test_eqodds_start, test_eqodds_end)
            test_eqodds_delta = abs(test_eqodds_end - test_eqodds_start)
            self._evolution["test_eqodds_rel"] = self._evolution["test_eqodds_rel"].apply(
                lambda x: (x - test_eqqods_min) / test_eqodds_delta
            )
        else:
            # Otherwise drop those columns.
            self._evolution.drop(
                ["test_eqodds", "test_eqodds_rel", "val_eqodds", "val_eqodds_rel"], axis=1, inplace=True
            )

        # If our goal was not fairness then the dataset bias is also not useful.
        if self.repairgoal != RepairGoal.FAIRNESS:
            self._evolution.drop("dataset_bias", axis=1, inplace=True)

        # Recompute relative accuracy and ROC AUC.
        val_accuracy_end = val_accuracy_utility_result.score
        val_accuracy_min = min(val_accuracy_start, val_accuracy_end)
        val_accuracy_delta = abs(val_accuracy_end - val_accuracy_start)
        self._evolution["val_accuracy_rel"] = self._evolution["val_accuracy_rel"].apply(
            lambda x: (x - val_accuracy_min) / val_accuracy_delta
        )
        test_accuracy_end = test_accuracy_utility_result.score
        test_accuracy_min = min(test_accuracy_start, test_accuracy_end)
        test_accuracy_delta = abs(test_accuracy_end - test_accuracy_start)
        self._evolution["test_accuracy_rel"] = self._evolution["test_accuracy_rel"].apply(
            lambda x: (x - test_accuracy_min) / test_accuracy_delta
        )
        val_roc_auc_end = val_roc_auc_utility_result.score
        val_roc_auc_min = min(val_roc_auc_start, val_roc_auc_end)
        val_roc_auc_delta = abs(val_roc_auc_end - val_roc_auc_start)
        self._evolution["val_roc_auc_rel"] = self._evolution["val_roc_auc_rel"].apply(
            lambda x: (x - val_roc_auc_min) / val_roc_auc_delta
        )
        test_roc_auc_end = test_roc_auc_utility_result.score
        test_roc_auc_min = min(test_roc_auc_start, test_roc_auc_end)
        test_roc_auc_delta = abs(test_roc_auc_end - test_roc_auc_start)
        self._evolution["test_roc_auc_rel"] = self._evolution["test_roc_auc_rel"].apply(
            lambda x: (x - test_roc_auc_min) / test_roc_auc_delta
        )

        # Close progress bar.
        if progress_bar:
            self.progress.close()
