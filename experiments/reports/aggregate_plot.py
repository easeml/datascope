import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from enum import Enum
from functools import partial
from matplotlib.figure import Figure
from typing import Callable, Optional, Dict, Any, Sequence, List

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

VALUE_MEASURE_C: Dict[AggregationMode, str] = {
    AggregationMode.MEAN_STD: "mean",
    AggregationMode.MEDIAN_PERC_90: "median",
    AggregationMode.MEDIAN_PERC_95: "median",
    AggregationMode.MEDIAN_PERC_99: "median",
}

DEFAULT_LABEL_FORMAT = "{compare}"


def filter(
    dataframe: DataFrame,
    attributes: Optional[Dict[str, Any]] = None,
) -> DataFrame:
    if attributes is None:
        return dataframe
    slicequery = " & ".join("%s == %s" % (str(k), repr(v)) for (k, v) in attributes.items())
    return dataframe.query(slicequery)


def aggregate(
    dataframe: DataFrame,
    targetval: str,
    index: str,
    compare: Optional[Sequence[str]] = None,
    aggmode: Optional[AggregationMode] = None,
) -> DataFrame:
    if aggmode is None:
        aggmode = AggregationMode.MEDIAN_PERC_95

    if compare is None:
        compare = []
    value_measures = VALUE_MEASURES[aggmode]

    groupbycols = [index] + list(compare)
    values = [
        dataframe.groupby(groupbycols)[[targetval]].agg(f).add_suffix(":" + k) for (k, f) in value_measures.items()
    ]
    dataframe = pd.concat(values, axis=1)
    dataframe.sort_index(inplace=True)
    if len(compare) > 0:
        dataframe = dataframe.unstack()
        dataframe.columns = dataframe.columns.swaplevel().map(">".join)
    return dataframe


def summarize(
    dataframe: DataFrame,
    summarize: str,
    compare: Optional[Sequence[str]] = None,
    summode: Optional[AggregationMode] = None,
) -> dict:
    if summode is None:
        summode = AggregationMode.MEAN_STD
    if compare is None:
        compare = []
    value_measures = VALUE_MEASURES[summode]
    values = [dataframe.groupby(compare)[[summarize]].agg(f).add_suffix(":" + k) for (k, f) in value_measures.items()]
    axis = 0 if len(compare) == 0 else 1
    dataframe = pd.concat(values, axis=axis)
    dataframe.sort_index(inplace=True)
    return dataframe.to_dict(orient="index")


def replace_keywords(source: str, keyword_replacements: Dict[str, str]) -> str:
    for k, v in sorted(keyword_replacements.items(), key=lambda x: len(x[1]), reverse=True):
        source = source.replace(k, v)
    return source


def plot(
    dataframe: DataFrame,
    index: str,
    targetval: str,
    compare: Optional[Sequence[str]] = None,
    aggmode: Optional[AggregationMode] = None,
    labelformat: Optional[str] = None,
    attributes: Optional[Dict[str, Any]] = None,
    summary: Optional[dict] = None,
    keyword_replacements: Optional[Dict[str, str]] = None,
) -> Figure:
    if aggmode is None:
        aggmode = AggregationMode.MEDIAN_PERC_95
    if attributes is None:
        attributes = {}
    if compare is None:
        compare = []
    if labelformat is None:
        if len(compare) > 0:
            labelformat = "; ".join("%%(%s)s" % c for c in compare)
        else:
            labelformat = targetval
    if keyword_replacements is None:
        keyword_replacements = {}

    comparison = []
    if len(compare) > 0:
        comparison = sorted(list(set(tuple(c.split(">")[:-1]) for c in dataframe.columns)))
        dataframe.columns = dataframe.columns.map(lambda x: (x.split(">")[0], x.split(":")[-1]))
        dataframe.sort_index(axis=1, inplace=True)
    else:
        comparison = [(targetval,)]
        dataframe.columns = dataframe.columns.map(lambda x: (targetval, x))

    figure = plt.figure(figsize=(10, 8))
    ax: plt.Axes = figure.subplots()

    for i, comp in enumerate(comparison):
        cols: List[str] = dataframe[comp].columns.to_list()
        uppercol = next(c for c in cols if c.endswith("-h"))
        lowercol = next(c for c in cols if c.endswith("-l"))
        upper = dataframe[comp][uppercol]
        lower = dataframe[comp][lowercol]
        ax.fill_between(dataframe.index.values, upper, lower, color=COLORS[i], alpha=0.2)

    for i, comp in enumerate(comparison):
        centercol = VALUE_MEASURE_C[aggmode]
        formatdict = dict(zip(compare, comp))
        if summary is not None:
            comp_summary = summary
            for c in comp:
                comp_summary = comp_summary.get(c, comp_summary)
            formatdict.update(comp_summary)
        label = labelformat % formatdict
        label = replace_keywords(label, keyword_replacements)
        ax.plot(dataframe[comp][centercol], color=COLORS[i], label=label)

    ax.set_title(" ".join("%s=%s" % (str(k), repr(v)) for (k, v) in attributes.items()))
    ax.set_xlim([dataframe.index.values[0], (dataframe.index.values[-1] - dataframe.index.values[0]) * 1.2])
    ax.set_ylim([-0.2, 1])
    ax.set_ylabel(targetval.title())
    ax.set_xlabel(index.title())
    ax.legend(loc="lower right")
    return figure


class AggregatePlot(Report, id="aggplot"):
    def __init__(
        self,
        study: Study,
        targetval: str,
        index: str,
        compare: Optional[Sequence[str]] = None,
        summarize: Optional[str] = None,
        errdisplay: Optional[ErrorDisplay] = None,
        aggmode: Optional[AggregationMode] = None,
        summode: Optional[AggregationMode] = None,
        groupby: Optional[Dict[str, Any]] = None,
        labelformat: Optional[str] = None,
        id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(study, id=id, groupby=groupby, **kwargs)
        self._targetval = targetval
        self._compare: Sequence[str] = compare if compare is not None else []
        self._index = index
        self._summarize = summarize
        self._errdisplay = errdisplay if errdisplay is not None else ErrorDisplay.SHADE
        self._aggmode = aggmode if aggmode is not None else AggregationMode.MEDIAN_PERC_95
        self._summode = summode if summode is not None else AggregationMode.MEAN_STD
        if labelformat is None:
            labelformat = "; ".join("%%(%s)s" % x for x in self._compare) if len(self._compare) > 0 else targetval
        self._labelformat = labelformat
        self._view: Optional[DataFrame] = None
        self._figure: Optional[Figure] = None
        self._summary: Optional[dict] = None

    @attribute
    def targetval(self) -> str:
        """The result attribute to evaluate."""
        return self._targetval

    @attribute
    def compare(self) -> Sequence[str]:
        """The attribute to compare by."""
        return self._compare

    @attribute
    def index(self) -> str:
        """The attribute to use as index."""
        return self._index

    @attribute
    def summarize(self) -> Optional[str]:
        """The attribute aggregate over the entire dataframe."""
        return self._summarize

    @attribute
    def aggmode(self) -> AggregationMode:
        """The mode of aggregation to apply."""
        return self._aggmode

    @attribute
    def summode(self) -> AggregationMode:
        """The mode of summarization to apply."""
        return self._summode

    @attribute
    def errdisplay(self) -> ErrorDisplay:
        """The method of displaying error bars."""
        return self._errdisplay

    @attribute
    def labelformat(self) -> str:
        """
        The string used to format labels in figures. A wildcard such as %%(attrname)s of %%(attrname).2f can be used.
        It will be replaced with a value where 'attrname' is either comparison attribute or a summary attribute.
        """
        return self._labelformat

    def generate(self) -> None:
        dataframe = filter(self.study.dataframe, attributes=self.groupby)

        self._view = aggregate(
            dataframe=dataframe,
            index=self._index,
            compare=self._compare,
            targetval=self._targetval,
            aggmode=self._aggmode,
        )
        if self._summarize:
            self._summary = summarize(
                dataframe=dataframe, summarize=self._summarize, compare=self._compare, summode=self._summode
            )

        keyword_replacements: Dict[str, str] = {}
        for scenario in self.study.scenarios:
            keyword_replacements.update(scenario.keyword_replacements)

        self._figure = plot(
            self._view,
            index=self._index,
            targetval=self._targetval,
            attributes=self._groupby,
            summary=self._summary,
            compare=self._compare,
            labelformat=self._labelformat,
            aggmode=self._aggmode,
            keyword_replacements=keyword_replacements,
        )

    @result
    def view(self) -> Optional[DataFrame]:
        return self._view

    @result
    def figure(self) -> Optional[Figure]:
        return self._figure

    @result
    def summary(self) -> Optional[dict]:
        return self._summary
