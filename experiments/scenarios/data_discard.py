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
from typing import Any, Iterable, Optional

from experiments.dataset.base import BiasedMixin

from .datascope_scenario import (
    DatascopeScenario,
    RepairMethod,
    IMPORTANCE_METHODS,
    MC_ITERATIONS,
    DEFAULT_SEED,
    DEFAULT_CHECKPOINTS,
    UtilityType,
)
from .base import attribute
from ..dataset import Dataset, DEFAULT_TRAINSIZE, DEFAULT_VALSIZE
from ..pipelines import Pipeline, get_model, ModelType


DEFAULT_TRAIN_BIAS = 0.8
DEFAULT_VAL_BIAS = 0.0


class DataDiscardScenario(DatascopeScenario, id="data-discard"):
    def __init__(
        self,
        dataset: str,
        pipeline: str,
        method: RepairMethod,
        utility: UtilityType,
        iteration: int,
        trainbias: float = DEFAULT_TRAIN_BIAS,
        valbias: float = DEFAULT_VAL_BIAS,
        seed: int = DEFAULT_SEED,
        trainsize: int = DEFAULT_TRAINSIZE,
        valsize: int = DEFAULT_VALSIZE,
        checkpoints: int = DEFAULT_CHECKPOINTS,
        evolution: Optional[pd.DataFrame] = None,
        importance_compute_time: Optional[float] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(
            dataset=dataset,
            pipeline=pipeline,
            method=method,
            utility=utility,
            iteration=iteration,
            seed=seed,
            trainsize=trainsize,
            valsize=valsize,
            checkpoints=checkpoints,
            evolution=evolution,
            importance_compute_time=importance_compute_time,
            **kwargs
        )
        self._trainbias = trainbias
        self._valbias = valbias

    @attribute
    def trainbias(self) -> float:
        """The bias of the training dataset used in fairness experiments."""
        return self._trainbias

    @attribute
    def valbias(self) -> float:
        """The bias of the validation dataset used in fairness experiments."""
        return self._valbias

    @classmethod
    def is_valid_config(cls, **attributes: Any) -> bool:
        result = True
        if "dataset" in attributes:
            dataset = Dataset.datasets[attributes["dataset"]]()
            result = result and isinstance(dataset, BiasedMixin)
        if "method" in attributes:
            result = result and not RepairMethod.is_tmc_nonpipe(attributes["method"])
        return result and super().is_valid_config(**attributes)

    def _run(self, progress_bar: bool = True, **kwargs: Any) -> None:

        # Load dataset.
        seed = self._seed + self._iteration
        dataset = Dataset.datasets[self.dataset](trainsize=self.trainsize, valsize=self.valsize, seed=seed)
        assert isinstance(dataset, BiasedMixin)
        dataset.load_biased(train_bias=self._trainbias)
        self.logger.debug(
            "Dataset '%s' loaded (trainsize=%d, valsize=%d).", self.dataset, dataset.trainsize, dataset.valsize
        )

        # Compute sensitive feature groupings.
        groupings = compute_groupings(dataset.X_val, dataset.sensitive_feature)

        # Create the dirty dataset and apply the data corruption.
        # dataset_dirty = deepcopy(dataset)
        # random = np.random.RandomState(seed=self._seed + self._iteration)
        # dirty_probs = [1 - self._dirty_ratio, self._dirty_ratio]
        # dirty_idx = random.choice(a=[False, True], size=(dataset_dirty.trainsize), p=dirty_probs)
        # assert dataset_dirty.y_train is not None
        # dataset_dirty.y_train[dirty_idx] = 1 - dataset_dirty.y_train[dirty_idx]

        # Load the pipeline and process the data.
        pipeline = Pipeline.pipelines[self.pipeline].construct(dataset)
        # X_train, y_train = dataset.X_train, dataset.y_train
        # X_val, y_val = dataset.X_val, dataset.y_val
        # X_train_orig, y_train_orig = dataset.X_train, dataset.y_train
        if RepairMethod.is_tmc_nonpipe(self.method):
            raise ValueError("This is not supported at the moment.")
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
        model = get_model(ModelType.LogisticRegression)
        # model_pipeline = deepcopy(pipeline)
        # pipeline.steps.append(("model", model))
        # if RepairMethod.is_pipe(self.method):
        #     model = model_pipeline
        accuracy_utility = SklearnModelAccuracy(model)
        eqodds_utility = SklearnModelEqualizedOddsDifference(
            model, sensitive_features=dataset.sensitive_feature, groupings=groupings
        )

        # target_model = model if RepairMethod.is_tmc_nonpipe(self.method) else pipeline
        target_utility: Utility
        if self.utility == UtilityType.ACCURACY:
            target_utility = JointUtility(SklearnModelAccuracy(model), weights=[-1.0])
        elif self.utility == UtilityType.EQODDS:
            target_utility = SklearnModelEqualizedOddsDifference(
                model, sensitive_features=dataset.sensitive_feature, groupings=groupings
            )
        elif self.utility == UtilityType.EQODDS_AND_ACCURACY:
            target_utility = JointUtility(
                SklearnModelAccuracy(model),
                SklearnModelEqualizedOddsDifference(
                    model, sensitive_features=dataset.sensitive_feature, groupings=groupings
                ),
                weights=[-0.5, 0.5],
            )
        else:
            raise ValueError("Unknown utility type '%s'." % repr(self.utility))

        # Compute importance scores and time it.
        random = np.random.RandomState(seed=seed)
        importance_time_start = process_time_ns()
        importance: Optional[ShapleyImportance] = None
        importances: Optional[Iterable[float]] = None
        if self.method == RepairMethod.RANDOM:
            importances = list(random.rand(dataset.trainsize))
        else:
            method = IMPORTANCE_METHODS[self.method]
            mc_iterations = MC_ITERATIONS[self.method]
            mc_preextract = RepairMethod.is_tmc_nonpipe(self.method)
            importance = ShapleyImportance(
                method=method,
                utility=target_utility,
                pipeline=pipeline,
                mc_iterations=mc_iterations,
                mc_timeout=self.timeout,
                mc_preextract=mc_preextract,
            )
            importances = importance.fit(dataset.X_train, dataset.y_train).score(dataset.X_val, dataset.y_val)
        importance_time_end = process_time_ns()
        self._importance_compute_time = (importance_time_end - importance_time_start) / 1e9
        self.logger.debug("Importance computed in: %s", str(timedelta(seconds=self._importance_compute_time)))
        n_units = dataset.trainsize
        discarded_units = np.zeros(n_units, dtype=bool)
        argsorted_importances = (-np.array(importances)).argsort()
        # argsorted_importances = np.ma.array(importances, mask=discarded_units).argsort()

        # Run the model to get initial score.
        dataset_f = dataset.apply(pipeline)
        eqodds = eqodds_utility(dataset_f.X_train, dataset_f.y_train, dataset_f.X_val, dataset_f.y_val)
        accuracy = accuracy_utility(dataset_f.X_train, dataset_f.y_train, dataset_f.X_val, dataset_f.y_val)

        # Update result table.
        evolution = [[0.0, eqodds, eqodds, accuracy, accuracy, 0, 0.0, dataset.train_bias, dataset.y_train.mean()]]
        eqodds_start = eqodds
        accuracy_start = accuracy

        # Set up progress bar.
        checkpoints = self.checkpoints if self.checkpoints > 0 and self.checkpoints < n_units else n_units
        n_units_per_checkpoint = round(self._trainbias * n_units / checkpoints)
        if progress_bar:
            self.progress.start(total=checkpoints, desc="(id=%s) Repairs" % self.id)
        # pbar = None if not progress_bar else tqdm(total=dataset.trainsize, desc="%s Repairs" % str(self))

        # Iterate over the repair process.
        # discarded_units = np.zeros(dataset.trainsize, dtype=bool)
        dataset_current = deepcopy(dataset)
        for i in range(checkpoints):

            # Determine indices of data examples that should be discarded given the unit with the highest importance.
            present_units = np.invert(discarded_units)
            target_units = argsorted_importances[present_units[argsorted_importances]]
            target_units = target_units[:n_units_per_checkpoint]
            discarded_units[target_units] = True
            present_units = np.invert(discarded_units).astype(int)
            present_idx = get_indices(provenance, present_units)
            dataset_current.X_train = dataset.X_train[present_idx, :]
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
            eqodds = eqodds_utility(
                dataset_current_f.X_train, dataset_current_f.y_train, dataset_current_f.X_val, dataset_current_f.y_val
            )
            accuracy = accuracy_utility(
                dataset_current_f.X_train, dataset_current_f.y_train, dataset_current_f.X_val, dataset_current_f.y_val
            )

            # Update result table.
            steps_rel = (i + 1) / float(checkpoints)
            discarded = discarded_units.sum(dtype=int)
            discarded_rel = discarded / float(n_units)
            dataset_bias = dataset_current.train_bias
            mean_label = np.mean(dataset_current.y_train)
            evolution.append(
                [steps_rel, eqodds, eqodds, accuracy, accuracy, discarded, discarded_rel, dataset_bias, mean_label]
            )

            # Recompute if needed.
            if importance is not None and self.method == RepairMethod.KNN_Interactive:
                importance_time_start = process_time_ns()
                importances = np.empty(n_units, dtype=float)
                importances[present_idx] = importance.fit(dataset_current.X_train, dataset_current.y_train).score(
                    dataset_current.X_val, dataset_current.y_val
                )
                importance_time_end = process_time_ns()
                self._importance_compute_time += (importance_time_end - importance_time_start) / 1e9
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
            ],
        )
        self._evolution.index.name = "steps"

        # Fix relative score.
        eqodds_end = eqodds
        eqqods_min = min(eqodds_start, eqodds_end)
        eqodds_delta = abs(eqodds_end - eqodds_start)
        accuracy_end = accuracy
        accuracy_min = min(accuracy_start, accuracy_end)
        accuracy_delta = abs(accuracy_end - accuracy_start)
        self._evolution["eqodds_rel"] = self._evolution["eqodds_rel"].apply(lambda x: (x - eqqods_min) / eqodds_delta)
        self._evolution["accuracy_rel"] = self._evolution["accuracy_rel"].apply(
            lambda x: (x - accuracy_min) / accuracy_delta
        )

        # Close progress bar.
        if progress_bar:
            self.progress.close()
