import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from enum import Enum
from functools import partial
from matplotlib.figure import Figure
from typing import Callable, Optional, Dict, Any, Sequence

from pandas import DataFrame

from ..scenarios import Report, Study, result, attribute


COLORS = ["#2BBFD9", "#FAC802", "#8AB365", "#DF362A", "#C670D2", "#C58F21", "#00AC9B", "#DC0030"]
LABELS = {
    "random": "Random",
    "shapley-tmc": "Shapley TMC",
    "shapley-knn-single": "Shapley KNN Single",
    "shapley-knn-interactive": "Shapley KNN Interactive",
    "shapley-tmc-10": "Shapley TMC x10",
    "shapley-tmc-50": "Shapley TMC x50",
    "shapley-tmc-100": "Shapley TMC x100",
    "shapley-tmc-500": "Shapley TMC x500",
}


class ErrorDisplay(str, Enum):
    BAR = "bar"
    SHADE = "shade"
    NONE = "none"


class AggregationMode(str, Enum):
    MEDIAN_PERC_90 = "median-perc-90"
    MEDIAN_PERC_95 = "median-perc-95"
    MEDIAN_PERC_99 = "median-perc-99"
    MEAN_STD = "mean-std"


VALUE_MEASURES: Dict[AggregationMode, Dict[str, Callable]] = {
    AggregationMode.MEAN_STD: {
        "mean": np.mean,
        "std": np.std,
        "std-l": lambda x: np.mean(x) - np.std(x),
        "std-h": lambda x: np.mean(x) + np.std(x),
    },
    AggregationMode.MEDIAN_PERC_90: {
        "median": np.median,
        "95perc-l": partial(np.percentile, q=5),
        "95perc-h": partial(np.percentile, q=95),
    },
    AggregationMode.MEDIAN_PERC_95: {
        "median": np.median,
        "95perc-l": partial(np.percentile, q=2.5),
        "95perc-h": partial(np.percentile, q=97.5),
    },
    AggregationMode.MEDIAN_PERC_99: {
        "median": np.median,
        "95perc-l": partial(np.percentile, q=0.5),
        "95perc-h": partial(np.percentile, q=99.5),
    },
}


def aggregate(
    dataframe: DataFrame,
    index: str,
    compare: str,
    value: str,
    aggmode: Optional[AggregationMode] = None,
    attributes: Optional[Dict[str, Any]] = None,
) -> DataFrame:
    if aggmode is None:
        aggmode = AggregationMode.MEDIAN_PERC_95
    if attributes is None:
        attributes = {}
    value_measures = VALUE_MEASURES[aggmode]

    slicequery = " & ".join("%s == %s" % (str(k), repr(v)) for (k, v) in attributes.items())
    dataframe = dataframe.query(slicequery)
    values = [
        dataframe.groupby([index] + [compare])[[value]].agg(f).add_suffix(":" + k) for (k, f) in value_measures.items()
    ]
    dataframe = pd.concat(values, axis=1)
    dataframe = dataframe.unstack()
    dataframe.columns = dataframe.columns.swaplevel().map(">".join)
    return dataframe


def plot(dataframe: DataFrame, index: str, value: str, attributes: Optional[Dict[str, Any]] = None) -> Figure:
    if attributes is None:
        attributes = {}

    compare = sorted(list(set(c.split(">")[0] for c in dataframe.columns)))
    idx = dataframe.index.values

    dataframe.columns = dataframe.columns.map(lambda x: (x.split(">")[0], x.split(":")[-1]))

    figure = plt.figure(figsize=(10, 8))
    ax: plt.Axes = figure.subplots()

    for i, comp in enumerate(compare):
        upper = dataframe[comp]["95perc-h"]
        lower = dataframe[comp]["95perc-l"]
        ax.fill_between(idx, upper, lower, color=COLORS[i], alpha=0.2)

    for i, comp in enumerate(compare):
        ax.plot(dataframe[comp]["median"], color=COLORS[i], label=LABELS[comp])

    ax.set_title(" ".join("%s=%s" % (str(k), repr(v)) for (k, v) in attributes.items()))
    ax.set_ylim([0, 1])
    ax.set_ylabel(value)
    ax.set_xlabel(index)
    ax.legend(loc="lower right")
    return figure


class AggregatePlot(Report, id="aggplot"):
    def __init__(
        self,
        study: Study,
        eval: str,
        compare: Optional[Sequence[str]] = None,
        index: Optional[str] = None,
        errdisplay: Optional[ErrorDisplay] = None,
        aggmode: Optional[AggregationMode] = None,
        groupby: Optional[Dict[str, Any]] = None,
        id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(study, id=id, groupby=groupby, **kwargs)
        self._eval = eval
        self._compare: Sequence[str] = compare if compare is not None else []
        self._index = index
        self._errdisplay = errdisplay if errdisplay is not None else ErrorDisplay.SHADE
        self._aggmode = aggmode if aggmode is not None else AggregationMode.MEDIAN_PERC_95
        self._view: Optional[DataFrame] = None
        self._figure: Optional[Figure] = None

    @attribute
    def eval(self) -> str:
        """The result attribute to evaluate."""
        return self._eval

    @attribute
    def compare(self) -> Sequence[str]:
        """The attribute to compare by."""
        return self._compare

    @attribute
    def index(self) -> Optional[str]:
        """The attribute to use as index."""
        return self._index

    @attribute
    def aggmode(self) -> AggregationMode:
        """The mode of aggregation to apply."""
        return self._aggmode

    @attribute
    def errdisplay(self) -> ErrorDisplay:
        """The method of displaying error bars."""
        return self._errdisplay

    def generate(self) -> None:
        self._view = aggregate(
            self.study.dataframe,
            index=self.index,
            compare=self.compare,
            value=self.eval,
            attributes=self.groupby,
            aggmode=self.aggmode,
        )
        self._figure = plot(self._view, index=self.index, value=self.eval, attributes=self.groupby)

    @result
    def view(self) -> Optional[DataFrame]:
        return self._view

    @result
    def figure(self) -> Optional[Figure]:
        return self._figure
