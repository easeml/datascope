from copy import deepcopy
import numpy as np
import pandas as pd

from datascope.importance.common import (
    SklearnModelAccuracy,
    JointUtility,
    SklearnModelEqualizedOddsDifference,
    Utility,
    compute_groupings,
)
from datascope.importance.shapley import ShapleyImportance
from datetime import timedelta
from time import process_time_ns
from typing import Any, Iterable, List, Optional, Union, Dict

from .base import attribute
from .datascope_scenario import (
    DatascopeScenario,
    RepairMethod,
    IMPORTANCE_METHODS,
    MC_ITERATIONS,
    DEFAULT_SEED,
    DEFAULT_CHECKPOINTS,
    DEFAULT_PROVIDERS,
    DEFAULT_MODEL,
    DEFAULT_REPAIR_GOAL,
    DEFAULT_TIMEOUT,
    UtilityType,
    RepairGoal,
)
from ..datasets import (
    Dataset,
    BiasMethod,
    BiasedDirtyLabelDataset,
    DirtyLabelDataset,
    DEFAULT_TRAINSIZE,
    DEFAULT_VALSIZE,
    DEFAULT_TESTSIZE,
)
from ..pipelines import Pipeline, ModelType, get_model


DEFAULT_DIRTY_RATIO = 0.5
DEFAULT_DIRTY_BIAS = 0.0
KEYWORD_REPLACEMENTS = {
    "accuracy": "Accuracy",
    "accuracy_rel": "Relative Accuracy",
    "repaired": "Number of Labels Examined",
    "repaired_rel": "Portion of Labels Examined",
    "discovered": "Number of Dirty Labels Found",
    "discovered_rel": "Portion of Dirty Labels Found",
}


class LabelRepairScenario(DatascopeScenario, id="label-repair"):
    def __init__(
        self,
        dataset: str,
        pipeline: str,
        method: RepairMethod,
        iteration: int,
        utility: UtilityType = UtilityType.ACCURACY,
        model: ModelType = DEFAULT_MODEL,
        dirtyratio: float = DEFAULT_DIRTY_RATIO,
        dirtybias: float = DEFAULT_DIRTY_BIAS,
        seed: int = DEFAULT_SEED,
        trainsize: int = DEFAULT_TRAINSIZE,
        valsize: int = DEFAULT_VALSIZE,
        testsize: int = DEFAULT_TESTSIZE,
        timeout: int = DEFAULT_TIMEOUT,
        checkpoints: int = DEFAULT_CHECKPOINTS,
        providers: int = DEFAULT_PROVIDERS,
        repairgoal: RepairGoal = DEFAULT_REPAIR_GOAL,
        evolution: Optional[pd.DataFrame] = None,
        importance_cputime: Optional[float] = None,
        **kwargs: Any
    ) -> None:
        kwargs.pop("biasmethod", None)
        super().__init__(
            dataset=dataset,
            pipeline=pipeline,
            method=method,
            utility=utility,
            iteration=iteration,
            model=model,
            seed=seed,
            trainsize=trainsize,
            valsize=valsize,
            testsize=testsize,
            timeout=timeout,
            checkpoints=checkpoints,
            providers=providers,
            repairgoal=repairgoal,
            biasmethod=BiasMethod.Feature,
            evolution=evolution,
            importance_cputime=importance_cputime,
            **kwargs
        )
        self._dirtyratio = dirtyratio
        self._dirtybias = dirtybias

    @classmethod
    def is_valid_config(cls, **attributes: Any) -> bool:
        result = True
        # if "method" in attributes:
        #     result = result and not RepairMethod.is_tmc_nonpipe(attributes["method"])
        if "dataset" in attributes:
            dataset_class = Dataset.datasets[attributes["dataset"]]
            result = result and issubclass(dataset_class, DirtyLabelDataset)

        if "repairgoal" not in attributes or attributes["repairgoal"] == RepairGoal.FAIRNESS:
            if "dataset" in attributes:
                dataset = Dataset.datasets[attributes["dataset"]]()
                result = result and isinstance(dataset, BiasedDirtyLabelDataset)
        elif "repairgoal" in attributes and attributes["repairgoal"] == RepairGoal.ACCURACY:
            if "dataset" in attributes:
                result = result and (attributes["dataset"] != "random")
            if "utility" in attributes:
                result = result and attributes["utility"] == UtilityType.ACCURACY
        return result and super().is_valid_config(**attributes)

    @attribute
    def dirtyratio(self) -> float:
        """The proportion of examples that will have corrupted labels in label repair experiments."""
        return self._dirtyratio

    @attribute
    def dirtybias(self) -> float:
        """The bias of dirty ratio between the sensitive data subgroup and the rest."""
        return self._dirtybias

    @property
    def keyword_replacements(self) -> Dict[str, str]:
        result = super().keyword_replacements
        return {**result, **KEYWORD_REPLACEMENTS}

    def _run(self, progress_bar: bool = True, **kwargs: Any) -> None:

        # Load dataset.
        seed = self._seed + self._iteration
        dataset = Dataset.datasets[self.dataset](
            trainsize=self.trainsize, valsize=self.valsize, testsize=self.testsize, seed=seed
        )
        assert isinstance(dataset, DirtyLabelDataset)
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
        if isinstance(dataset, BiasedDirtyLabelDataset):
            groupings_val = compute_groupings(dataset.X_val, dataset.sensitive_feature)
            groupings_test = compute_groupings(dataset.X_test, dataset.sensitive_feature)

        # Create the dirty dataset and apply the data corruption.
        # dataset_dirty = deepcopy(dataset)
        # random = np.random.RandomState(seed=self._seed + self._iteration)
        # dirty_probs = [1 - self._dirtyratio, self._dirtyratio]
        # dirty_idx = random.choice(a=[False, True], size=(dataset_dirty.trainsize), p=dirty_probs)
        # dataset_dirty.y_train[dirty_idx] = 1 - dataset_dirty.y_train[dirty_idx]
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
            assert isinstance(dataset, BiasedDirtyLabelDataset)
            dataset_dirty = dataset.corrupt_labels_with_bias(probabilities=probabilities, groupbias=self.dirtybias)
            units_dirty = deepcopy(dataset_dirty.units_dirty)

        # Load the pipeline and process the data.
        pipeline = Pipeline.pipelines[self.pipeline].construct(dataset)
        # X_train, y_train = dataset.X_train, dataset.y_train
        # X_val, y_val = dataset.X_val, dataset.y_val
        # X_train_dirty, y_train_dirty = dataset_dirty.X_train, dataset_dirty.y_train
        # assert X_train is not None and y_train is not None
        # assert X_train_dirty is not None and y_train_dirty is not None
        # assert X_val is not None and y_val is not None
        # if RepairMethod.is_tmc_nonpipe(self.method):
        #     raise ValueError("This is not supported at the moment.")
        #     self.logger.debug("Shape of X_train before feature extraction: %s", str(X_train.shape))
        #     X_train = pipeline.fit_transform(X_train, y_train)  # TODO: Fit the pipeline with dirty data.
        #     assert isinstance(X_train, ndarray)
        #     X_train_dirty = pipeline.transform(X_train_dirty)
        #     assert isinstance(X_train_dirty, ndarray)
        #     X_val = pipeline.transform(X_val)
        #     assert isinstance(X_val, ndarray)
        #     self.logger.debug("Shape of X_train after feature extraction: %s", str(X_train.shape))

        # Reshape datasets if needed.
        # if X_train.ndim > 2:
        #     X_train = X_train.reshape(X_train.shape[0], -1)
        #     X_train_dirty = X_train_dirty.reshape(X_train_dirty.shape[0], -1)
        #     X_val = X_val.reshape(X_val.shape[0], -1)
        #     self.logger.debug("Need to reshape. New shape: %s", str(X_train.shape))

        # Construct binarized provenance matrix.
        # provenance = np.expand_dims(np.arange(dataset.trainsize, dtype=int), axis=(1, 2, 3))
        # provenance = np.pad(provenance, pad_width=((0, 0), (0, 0), (0, 0), (0, 1)))
        # provenance = binarize(provenance)

        # Initialize the model and utility.
        model = get_model(self.model)
        # model_pipeline = deepcopy(pipeline)
        # pipeline.steps.append(("model", model))
        # if RepairMethod.is_pipe(self.method):
        #     model = model_pipeline
        accuracy_utility = SklearnModelAccuracy(model)
        eqodds_utility: Optional[SklearnModelEqualizedOddsDifference] = None
        if self.repairgoal == RepairGoal.FAIRNESS:
            assert isinstance(dataset, BiasedDirtyLabelDataset)
            eqodds_utility = SklearnModelEqualizedOddsDifference(
                model, sensitive_features=dataset.sensitive_feature, groupings=groupings_test
            )

        # target_model = model if RepairMethod.is_tmc_nonpipe(self.method) else pipeline
        target_utility: Utility
        if self.utility == UtilityType.ACCURACY:
            target_utility = JointUtility(SklearnModelAccuracy(model), weights=[-1.0])
            # target_utility = SklearnModelAccuracy(model)
        elif self.utility == UtilityType.EQODDS:
            assert self.repairgoal == RepairGoal.FAIRNESS and isinstance(dataset, BiasedDirtyLabelDataset)
            target_utility = SklearnModelEqualizedOddsDifference(
                model, sensitive_features=dataset.sensitive_feature, groupings=groupings_val
            )
        elif self.utility == UtilityType.EQODDS_AND_ACCURACY:
            assert self.repairgoal == RepairGoal.FAIRNESS and isinstance(dataset, BiasedDirtyLabelDataset)
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
        importance_time_start = process_time_ns()
        n_units = dataset_dirty.units.shape[0]
        importance: Optional[ShapleyImportance] = None
        importances: Optional[Iterable[float]] = None
        random = np.random.RandomState(seed=self._seed + self._iteration + 1)
        if self.method == RepairMethod.RANDOM:
            importances = list(random.rand(n_units))
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
                logger=self.logger,
            )
            importances = importance.fit(
                dataset_dirty.X_train, dataset_dirty.y_train, provenance=dataset_dirty.provenance
            ).score(dataset.X_val, dataset.y_val)
        importance_time_end = process_time_ns()
        importance_cputime = (importance_time_end - importance_time_start) / 1e9
        self.logger.debug("Importance computed in: %s", str(timedelta(seconds=importance_cputime)))
        visited_units = np.zeros(n_units, dtype=bool)
        argsorted_importances = (-np.array(importances)).argsort()
        # argsorted_importances = np.ma.array(importances, mask=visited_units).argsort()

        # Run the model to get initial score.
        # assert y_val is not None
        dataset_f = dataset.apply(pipeline)
        dataset_dirty_f = dataset_dirty.apply(pipeline)
        eqodds: Optional[float] = None
        if eqodds_utility is not None:
            eqodds = eqodds_utility(
                dataset_dirty_f.X_train, dataset_dirty_f.y_train, dataset_f.X_test, dataset_f.y_test
            )
        accuracy = accuracy_utility(
            dataset_dirty_f.X_train, dataset_dirty_f.y_train, dataset_f.X_test, dataset_f.y_test
        )

        # Update result table.
        evolution = [[0.0, eqodds, eqodds, accuracy, accuracy, 0, 0.0, 0, 0.0, 0.0]]
        eqodds_start = eqodds
        accuracy_start = accuracy

        # Set up progress bar.
        checkpoints = self.checkpoints if self.checkpoints > 0 and self.checkpoints < n_units else n_units
        n_units_per_checkpoint = round(n_units / checkpoints)
        if progress_bar:
            self.progress.start(total=checkpoints, desc="(id=%s) Repairs" % self.id)
        # pbar = None if not progress_bar else tqdm(total=dataset.trainsize, desc="%s Repairs" % str(self))

        # Iterate over the repair process.
        # visited_units = np.zeros(dataset.trainsize, dtype=bool)
        for i in range(checkpoints):

            # Determine indices of data examples that should be repaired given the unit with the highest importance.
            unvisited_units = np.invert(visited_units)
            target_units = argsorted_importances[unvisited_units[argsorted_importances]]
            if i + 1 < checkpoints:
                target_units = target_units[:n_units_per_checkpoint]
            # target_query = np.zeros(n_units, dtype=int)
            # target_query[target_units] = 1
            # target_unit = np.ma.array(importances, mask=visited_units).argmin()
            # target_query = np.eye(1, visited_units.shape[0], target_unit, dtype=int).flatten()
            # target_idx = get_indices(dataset_dirty.provenance, target_query)

            # Repair the data example.
            # dataset_dirty.y_train[target_idx] = dataset.y_train[target_idx]
            visited_units[target_units] = True
            dataset_dirty.units_dirty[target_units] = False

            # Run the model.
            if eqodds_utility is not None:
                eqodds = eqodds_utility(
                    dataset_dirty_f.X_train,
                    dataset_dirty.y_train,
                    dataset_dirty_f.X_test,
                    dataset_dirty_f.y_test,
                )
            accuracy = accuracy_utility(
                dataset_dirty_f.X_train, dataset_dirty.y_train, dataset_dirty_f.X_test, dataset_dirty_f.y_test
            )

            # self.logger.debug("Dirty units: %.2f", np.sum(dataset_dirty.units_dirty))
            # self.logger.debug("Same labels: %.2f", np.sum(dataset_dirty.y_train == dataset.y_train))

            # Update result table.
            steps_rel = (i + 1) / float(checkpoints)
            repaired = visited_units.sum(dtype=int)
            repaired_rel = repaired / float(n_units)
            discovered = np.logical_and(visited_units, units_dirty).sum(dtype=int)
            discovered_rel = discovered / units_dirty.sum(dtype=float)
            evolution.append(
                [
                    steps_rel,
                    eqodds,
                    eqodds,
                    accuracy,
                    accuracy,
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
                    dataset_dirty.X_train, dataset_dirty.y_train, provenance=dataset_dirty.provenance
                ).score(dataset.X_val, dataset.y_val)
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
                "eqodds",
                "eqodds_rel",
                "accuracy",
                "accuracy_rel",
                "repaired",
                "repaired_rel",
                "discovered",
                "discovered_rel",
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

        # Recompute relative accuracy.
        accuracy_end = accuracy
        accuracy_delta = accuracy_end - accuracy_start
        self._evolution["accuracy_rel"] = self._evolution["accuracy_rel"].apply(
            lambda x: (x - accuracy_start) / accuracy_delta
        )

        # Close progress bar.
        if progress_bar:
            self.progress.close()
