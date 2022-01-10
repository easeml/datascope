from .common import Experiment, attribute
from ..datasets import DatasetId
from ..pipelines import PipelineId

from pandas import DataFrame
from typing import Any


class LabelRepairExperiment(Experiment, scenario="label-repair"):
    def __init__(self, dataset: DatasetId, pipeline: PipelineId, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._dataset = dataset
        self._pipeline = pipeline

    @attribute
    def dataset(self) -> DatasetId:
        return self._dataset

    @attribute
    def pipeline(self) -> PipelineId:
        return self._pipeline

    def run(self, **kwargs: Any) -> None:
        raise NotImplementedError()

    def completed(self) -> bool:
        raise NotImplementedError()

    def dataframe(self) -> DataFrame:
        raise NotImplementedError()
