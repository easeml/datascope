from copy import deepcopy
import numpy as np
import pandas as pd

from datascope.importance.common import (
    JointUtility,
    SklearnModelAccuracy,
    SklearnModelEqualizedOddsDifference,
    Utility,
    binarize,
    get_indices,
    compute_groupings,
)
from datascope.importance.shapley import ShapleyImportance
from datetime import timedelta
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
    DEFAULT_TRAINSIZE,
    DEFAULT_VALSIZE,
    DEFAULT_TESTSIZE,
    DEFAULT_BIAS_METHOD,
)
from ..pipelines import Pipeline, get_model


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
                result = result and attributes["utility"] == UtilityType.ACCURACY

        return result and super().is_valid_config(**attributes)

    def _run(self, progress_bar: bool = True, **kwargs: Any) -> None:

        # Load dataset.
        seed = self._seed + self._iteration
        dataset = Dataset.datasets[self.dataset](
            trainsize=self.trainsize, valsize=self.valsize, testsize=self.testsize, seed=seed
        )
        if self.repairgoal == RepairGoal.FAIRNESS:
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

        # Compute sensitive feature groupings.
        groupings_val: Optional[np.ndarray] = None
        groupings_test: Optional[np.ndarray] = None
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

        # Construct binarized provenance matrix.
        provenance = np.expand_dims(np.arange(dataset.trainsize, dtype=int), axis=(1, 2, 3))
        provenance = np.pad(provenance, pad_width=((0, 0), (0, 0), (0, 0), (0, 1)))
        provenance = binarize(provenance)

        # Initialize the model and utility.
        model_type = MODEL_TYPES[self.model]
        model_kwargs = MODEL_KWARGS[self.model]
        model = get_model(model_type, **model_kwargs)
        # model_pipeline = deepcopy(pipeline)
        # pipeline.steps.append(("model", model))
        # if RepairMethod.is_pipe(self.method):
        #     model = model_pipeline
        accuracy_utility = SklearnModelAccuracy(model)
        eqodds_utility: Optional[SklearnModelEqualizedOddsDifference] = None
        if self.repairgoal == RepairGoal.FAIRNESS:
            assert isinstance(dataset, BiasedMixin)
            eqodds_utility = SklearnModelEqualizedOddsDifference(
                model, sensitive_features=dataset.sensitive_feature, groupings=groupings_test
            )

        # target_model = model if RepairMethod.is_tmc_nonpipe(self.method) else pipeline
        target_utility: Utility
        if self.utility == UtilityType.ACCURACY:
            target_utility = JointUtility(SklearnModelAccuracy(model), weights=[-1.0])
            # target_utility = SklearnModelAccuracy(model)
        elif self.utility == UtilityType.EQODDS:
            assert self.repairgoal == RepairGoal.FAIRNESS and isinstance(dataset, BiasedMixin)
            target_utility = SklearnModelEqualizedOddsDifference(
                model, sensitive_features=dataset.sensitive_feature, groupings=groupings_val
            )
        elif self.utility == UtilityType.EQODDS_AND_ACCURACY:
            assert self.repairgoal == RepairGoal.FAIRNESS and isinstance(dataset, BiasedMixin)
            target_utility = JointUtility(
                SklearnModelAccuracy(model),
                SklearnModelEqualizedOddsDifference(
                    model, sensitive_features=dataset.sensitive_feature, groupings=groupings_val
                ),
                weights=[-0.5, 0.5],
            )
        else:
            raise ValueError("Unknown utility type '%s'." % repr(self.utility))

        # Compute importance scores and time it.
        random = np.random.RandomState(seed=seed)
        importance_time_start = process_time_ns()
        importance: Optional[ShapleyImportance] = None
        importances: Optional[np.ndarray] = None
        if self.method == RepairMethod.RANDOM:
            importances = np.array(random.rand(dataset.trainsize))
        else:
            method = IMPORTANCE_METHODS[self.method]
            mc_iterations = MC_ITERATIONS[self.method]
            mc_preextract = RepairMethod.is_tmc_nonpipe(self.method)
            importance = ShapleyImportance(
                method=method,
                utility=target_utility,
                pipeline=pipeline,
                mc_iterations=mc_iterations,
                mc_timeout=self.mc_timeout,
                mc_tolerance=self.mc_tolerance,
                mc_preextract=mc_preextract,
                nn_k=self.nn_k,
            )
            importances = np.array(importance.fit(dataset.X_train, dataset.y_train).score(dataset.X_val, dataset.y_val))
        importance_time_end = process_time_ns()
        importance_cputime = (importance_time_end - importance_time_start) / 1e9
        self.logger.debug("Importance computed in: %s", str(timedelta(seconds=importance_cputime)))
        n_units = dataset.trainsize
        discarded_units = np.zeros(n_units, dtype=bool)
        argsorted_importances = (-np.array(importances)).argsort()
        # argsorted_importances = np.ma.array(importances, mask=discarded_units).argsort()

        # Run the model to get initial score.
        dataset_f = dataset.apply(pipeline)
        eqodds: Optional[float] = None
        if eqodds_utility is not None:
            eqodds = eqodds_utility(dataset_f.X_train, dataset_f.y_train, dataset_f.X_test, dataset_f.y_test)
        accuracy = accuracy_utility(dataset_f.X_train, dataset_f.y_train, dataset_f.X_test, dataset_f.y_test)

        # Update result table.
        dataset_bias = dataset.train_bias if isinstance(dataset, BiasedMixin) else None
        evolution = [[0.0, eqodds, eqodds, accuracy, accuracy, 0, 0.0, dataset_bias, dataset.y_train.mean(), 0.0]]
        eqodds_start = eqodds
        accuracy_start = accuracy

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
            present_idx = get_indices(provenance, present_units)
            dataset_current.X_train = dataset.X_train[present_idx]
            dataset_current.y_train = dataset.y_train[present_idx]

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
            dataset_current_f = dataset_current.apply(pipeline)
            if eqodds_utility is not None:
                eqodds = eqodds_utility(
                    dataset_current_f.X_train,
                    dataset_current_f.y_train,
                    dataset_current_f.X_test,
                    dataset_current_f.y_test,
                )
            accuracy = accuracy_utility(
                dataset_current_f.X_train, dataset_current_f.y_train, dataset_current_f.X_test, dataset_current_f.y_test
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
                    eqodds,
                    eqodds,
                    accuracy,
                    accuracy,
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
                importances[present_idx] = importance.fit(dataset_current.X_train, dataset_current.y_train).score(
                    dataset_current.X_val, dataset_current.y_val
                )
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
                "eqodds",
                "eqodds_rel",
                "accuracy",
                "accuracy_rel",
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
            assert eqodds is not None and eqodds_start is not None
            eqodds_end = eqodds
            eqqods_min = min(eqodds_start, eqodds_end)
            eqodds_delta = abs(eqodds_end - eqodds_start)
            self._evolution["eqodds_rel"] = self._evolution["eqodds_rel"].apply(
                lambda x: (x - eqqods_min) / eqodds_delta
            )
        else:
            # Otherwise drop those columns.
            self._evolution.drop(["eqodds", "eqodds_rel"], axis=1, inplace=True)

        # If our goal was not fairness then the dataset bias is also not useful.
        if self.repairgoal != RepairGoal.FAIRNESS:
            self._evolution.drop("dataset_bias", axis=1, inplace=True)

        # Recompute relative accuracy.
        accuracy_end = accuracy
        accuracy_min = min(accuracy_start, accuracy_end)
        accuracy_delta = abs(accuracy_end - accuracy_start)
        self._evolution["accuracy_rel"] = self._evolution["accuracy_rel"].apply(
            lambda x: (x - accuracy_min) / accuracy_delta
        )

        # Close progress bar.
        if progress_bar:
            self.progress.close()
