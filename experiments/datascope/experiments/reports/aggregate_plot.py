import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re

from enum import Enum
from functools import partial
from itertools import combinations, product
from matplotlib.figure import Figure
from matplotlib.ticker import EngFormatter, PercentFormatter
from matplotlib.transforms import Bbox
from numpy.typing import NDArray
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import connected_components
from typing import Callable, Optional, Dict, Any, List, Union, TypeVar, Tuple

from pandas import DataFrame, MultiIndex
from pandas.core.groupby.generic import DataFrameGroupBy
from pandas.core.groupby.groupby import GroupBy

from ..bench import Report, Study, result, attribute

COLOR_NAMES = ["blue", "red", "yellow", "green", "purple", "brown", "cyan", "pink"]
COLORS = ["#2BBFD9", "#DF362A", "#FAC802", "#8AB365", "#C670D2", "#AE7E1E", "#008F6B", "#6839CC"]
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


class PlotType(str, Enum):
    BAR = "bar"
    LINE = "line"
    DOT = "dot"


class TickFormat(str, Enum):
    DEFAULT = "default"
    ENG = "engineer"
    PERCENT = "percent"


class AggregationMode(str, Enum):
    MEDIAN_PERC_90 = "median-perc-90"
    MEDIAN_PERC_95 = "median-perc-95"
    MEDIAN_PERC_99 = "median-perc-99"
    MEAN_STD = "mean-std"


VALUE_MEASURES: Dict[AggregationMode, Dict[str, Union[Callable, str]]] = {
    AggregationMode.MEAN_STD: {
        "mean": "mean",
        "std": "std",
        "std-l": lambda x: np.mean(x) - np.std(x),
        "std-h": lambda x: np.mean(x) + np.std(x),
    },
    AggregationMode.MEDIAN_PERC_90: {
        "median": "median",
        "95perc-l": partial(np.percentile, q=5),
        "95perc-h": partial(np.percentile, q=95),
    },
    AggregationMode.MEDIAN_PERC_95: {
        "median": "median",
        "95perc-l": partial(np.percentile, q=2.5),
        "95perc-h": partial(np.percentile, q=97.5),
    },
    AggregationMode.MEDIAN_PERC_99: {
        "median": "median",
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


class CrossAggregationMode(str, Enum):
    MEAN_SQUARE_ERROR = "mse"
    ROOT_MEAN_SQUARE_ERROR = "rmse"
    MEAN_ABSOLUTE_ERROR = "mae"
    MEAN_ABSOLUTE_PERCENTAGE_ERROR = "mape"


CROSS_AGGREGATION_FUNCTIONS: Dict[CrossAggregationMode, Callable[[NDArray, NDArray], float]] = {
    CrossAggregationMode.MEAN_SQUARE_ERROR: lambda x1, x2: np.square(x1 - x2).mean(),
    CrossAggregationMode.ROOT_MEAN_SQUARE_ERROR: lambda x1, x2: np.sqrt(np.square(x1 - x2).mean()),
    CrossAggregationMode.MEAN_ABSOLUTE_ERROR: lambda x1, x2: np.abs(x1 - x2).mean(),
    CrossAggregationMode.MEAN_ABSOLUTE_PERCENTAGE_ERROR: lambda x1, x2: np.abs((x1 - x2) / (x1 + 1e-6)).mean(),
}


class SliceOp(str, Enum):
    COUNT = "count"
    FIRST = "first"
    LAST = "last"
    MAX = "max"
    MIN = "min"
    MEAN = "mean"
    STD = "std"
    MEDIAN = "median"


SLICE_OPS: Dict[SliceOp, Callable[[GroupBy], Any]] = {
    SliceOp.COUNT: GroupBy.count,
    SliceOp.FIRST: GroupBy.first,
    SliceOp.LAST: GroupBy.last,
    SliceOp.MAX: GroupBy.max,
    SliceOp.MIN: GroupBy.min,
    SliceOp.MEAN: GroupBy.mean,
    SliceOp.STD: GroupBy.std,
    SliceOp.MEDIAN: GroupBy.median,
}

DEFAULT_LABEL_FORMAT = "{compare}"
DEFAULT_FONTSIZE = 16
DEFAULT_LINEWIDTH = 2.5


def represent(x: Any):
    if isinstance(x, Enum):
        return repr(x.value)
    if isinstance(x, float):
        return "%.2f" % x
    else:
        return repr(x)


def filter(
    dataframe: DataFrame,
    attributes: Optional[Dict[str, Any]] = None,
    resultfilter: Optional[str] = None,
) -> DataFrame:
    if attributes is None or len(attributes) == 0:
        return dataframe
    slicequery = " & ".join("%s == %s" % (str(k), represent(v)) for (k, v) in attributes.items())
    if resultfilter is not None and len(resultfilter) > 0:
        slicequery += " & " + resultfilter
    return dataframe.query(slicequery)


def aggregate(
    dataframe: DataFrame,
    index: str,
    targetval: List[str],
    compare: Optional[List[str]] = None,
    aggmode: AggregationMode = AggregationMode.MEDIAN_PERC_95,
) -> DataFrame:
    if compare is None:
        compare = []
    value_measures = VALUE_MEASURES[aggmode]

    groupbycols = [index] + list(compare)
    dataframe = dataframe[[index] + targetval + compare].dropna()
    values = [dataframe.groupby(groupbycols)[targetval].agg(f).add_suffix(":" + k) for (k, f) in value_measures.items()]
    dataframe = pd.concat(values, axis=1)
    dataframe.sort_index(inplace=True)
    if len(compare) > 0:
        unstacked = dataframe.unstack()
        dataframe = unstacked if isinstance(unstacked, DataFrame) else unstacked.to_frame()
        assert isinstance(dataframe.columns, MultiIndex)
        dataframe.columns = dataframe.columns.swaplevel().map(lambda x: tuple(str(xx) for xx in x)).map(">".join)
    return dataframe


def summarize(
    dataframe: DataFrame,
    summarize: List[str],
    compare: Optional[List[str]] = None,
    summode: Optional[AggregationMode] = None,
) -> dict:
    if summode is None:
        summode = AggregationMode.MEAN_STD
    if compare is None:
        compare = []
    value_measures = VALUE_MEASURES[summode]
    dataframe = dataframe[summarize + compare].dropna()
    if len(compare) > 0:
        summary_frames = [
            dataframe.groupby(compare)[summarize].agg(f).add_suffix(":" + k) for (k, f) in value_measures.items()
        ]
        return pd.concat(summary_frames, axis=1).to_dict(orient="index")
    else:
        summary_series = [dataframe[summarize].agg(f).add_suffix(":" + k) for (k, f) in value_measures.items()]
        return pd.concat(summary_series, axis=0).to_dict()


def cross_aggregate(
    dataframe: pd.DataFrame,
    index: Optional[List[str]],
    compare: List[str],
    targetval: List[str],
    crossaggover: List[str],
    crossaggpairs: Optional[List[str]] = None,
    crossaggmode: CrossAggregationMode = CrossAggregationMode.MEAN_ABSOLUTE_ERROR,
):
    # Pivot the table so that target values from different comparison values appear side by side.
    index = [] if index is None else index
    crossaggover = [x for x in crossaggover if x not in index]
    assert index is not None
    indexlen = len(index)
    dataframe = dataframe.pivot(index=index + crossaggover, columns=compare, values=targetval)

    # Produce a series of pairwise comparisons for each target value and then concatdaenate them into one dataframe.
    method = CROSS_AGGREGATION_FUNCTIONS[crossaggmode]
    correlation_parts = []
    for val in targetval:
        target: Union[DataFrame, DataFrameGroupBy] = dataframe[[val]]
        if len(index) > 0:
            assert isinstance(target, DataFrame)
            target = target.groupby(index)
        target = target.corr(method=method)  # type: ignore
        target = target.stack(dropna=False, level=compare)  # type: ignore
        target = target.droplevel(axis=0, level=1 if len(index) > 0 else 0)  # type: ignore
        correlation_parts.append(target)
    dataframe = pd.concat(correlation_parts, axis=1)

    # Keep only a single instance of each pair.
    if crossaggpairs is None or len(crossaggpairs) == 0:
        comparevals = [sorted(dataframe[col].unique()) for col in compare]
        crossaggpairs = [x[0] + x[1] for x in combinations(product(*comparevals), r=2)]
    pairs = [tuple(x.replace("&", ",").split(",")) for x in crossaggpairs]
    dataframe = dataframe[dataframe.index.map(lambda x: x[indexlen:] in pairs)]

    # Move values from the index into regular columns.
    n = len(compare)

    def column_mapper(x: Tuple) -> Tuple:
        result = tuple(x[:indexlen])
        result += (",".join(str(x[i]) for i in range(-2 * n, -n)) + "&" + ",".join(str(x[i]) for i in range(-n, 0)),)
        result += tuple([None for _ in range(2 * n - 1)])
        return result

    dataframe.index = dataframe.index.map(column_mapper)
    dataframe.index = dataframe.index.droplevel(list(range(-2 * n + 1, 0)))
    dataframe.index = dataframe.index.set_names(">".join(compare), level=-1 if len(index) > 0 else None)
    dataframe = dataframe.reset_index()
    return dataframe


def replace_keywords(source: str, keyword_replacements: Dict[str, str]) -> str:
    for k, v in sorted(keyword_replacements.items(), key=lambda x: len(x[0]), reverse=True):
        source = re.sub("(?<![a-zA-Z])%s(?![a-z-Z])" % k, v, source)
    return source.replace("_", " ").title()


def get_colors(keys: List[Tuple[str, ...]], colors: Optional[Dict[Tuple[str, ...], str]]) -> List[str]:
    available_default_colors = [
        COLORS[i]
        for i in range(len(COLORS))
        if colors is None or (COLORS[i] not in colors.values() and COLOR_NAMES[i] not in colors.values())
    ]
    result_colors: List[str] = []
    for key in keys:
        if colors is not None and key in colors:
            color = colors[key]
            if color in COLOR_NAMES:
                color = COLORS[COLOR_NAMES.index(color)]
            result_colors.append(color)
        else:
            result_colors.append(available_default_colors.pop(0))
    return result_colors


def fix_text_positioning(texts: Union[List[plt.Text], List[plt.Annotation]], axes: plt.Axes) -> None:

    # Get renderer from axes.
    renderer = axes.figure.canvas.renderer  # type: ignore

    # Extract bounding boxes of all text objects.
    transforms = [text.get_transform() for text in texts]
    bboxes = [text.get_window_extent(renderer) for text in texts]
    # bboxes = [axes.transData.inverted().transform_bbox(bbox) for bbox in bboxes]
    heights = np.array([bbox.y1 - bbox.y0 for bbox in bboxes])
    vertical_positions = np.array([bbox.y0 for bbox in bboxes])

    # # Draw a bounding rectangle of all objects using bboxes.
    # for bbox in bboxes:
    #     rect_bbox = axes.transData.inverted().transform_bbox(bbox)
    #     axes.add_patch(
    #         Rectangle(
    #             (rect_bbox.xmin, rect_bbox.ymin),
    #             rect_bbox.width,
    #             rect_bbox.height,
    #             fill=False,
    #             edgecolor="red",
    #             # transform=axes.transAxes,
    #         )
    #     )

    # Find groups of texts that overlap.
    overlaps = lil_matrix((len(bboxes), len(bboxes)), dtype=bool)
    for i in range(len(bboxes)):
        for j in range(i + 1, len(bboxes)):
            if bboxes[i].overlaps(bboxes[j]):
                overlaps[i, j] = True
    n_components, labels = connected_components(csgraph=overlaps, directed=False, return_labels=True)
    for i in range(n_components):
        group = np.where(labels == i)[0]
        if len(group) > 1:
            # Reorder the group by vertical position.
            group = group[np.argsort(vertical_positions[group])]

            # Find the bounding box that covers the group and use it to adjust the position of the texts.
            bbox = Bbox.union([bboxes[i] for i in group])
            y_cur = (bbox.ymin + bbox.ymax) / 2 - np.sum(heights[group]) / 2 + heights[group[0]] / 2
            for i in group:
                text = texts[i]
                _, y = transforms[i].inverted().transform((0, y_cur))
                text.set_y(y)
                y_cur += heights[i]

    # Expand the axes to fit the new text positions.
    bboxes = [text.get_window_extent(renderer) for text in texts]
    bbox_union = Bbox.union(bboxes)
    bbox_data_coords = axes.transData.inverted().transform_bbox(bbox_union)
    # axes.update_datalim(bbox_union)
    if bbox_data_coords.xmin < axes.get_xlim()[0]:
        axes.set_xlim(xmin=bbox_data_coords.xmin)
    if bbox_data_coords.xmax > axes.get_xlim()[1]:
        axes.set_xlim(xmax=bbox_data_coords.xmax)
    if bbox_data_coords.ymin < axes.get_ylim()[0]:
        axes.set_ylim(ymin=bbox_data_coords.ymin)
    if bbox_data_coords.ymax > axes.get_ylim()[1]:
        axes.set_ylim(ymax=bbox_data_coords.ymax)


def lineplot(
    dataframe: DataFrame,
    index: str,
    targetval: str,
    compare: Optional[List[str]] = None,
    compareorder: List[str] = [],
    colors: Optional[Dict[str, str]] = None,
    aggmode: AggregationMode = AggregationMode.MEDIAN_PERC_95,
    errdisplay: ErrorDisplay = ErrorDisplay.SHADE,
    labelformat: Optional[str] = None,
    summary: Optional[dict] = None,
    keyword_replacements: Optional[Dict[str, str]] = None,
    axes: Optional[plt.Axes] = None,
    fontsize: int = DEFAULT_FONTSIZE,
    annotations: bool = False,
    dontcompare: Optional[str] = None,
    linemarker: Optional[str] = None,
) -> Optional[Figure]:
    if compare is None:
        compare = []
    if labelformat is None:
        if len(compare) > 0:
            labelformat = "; ".join("%%(%s)s" % c for c in compare)
        else:
            labelformat = targetval
    if keyword_replacements is None:
        keyword_replacements = {}
    if dontcompare is None:
        dontcompare = ""

    comparison = []
    dataframe = dataframe.copy()
    if len(compare) > 0:
        comparison = sorted(list(set(tuple(c.split(">")[:-1]) for c in dataframe.columns)))
        dataframe.columns = dataframe.columns.map(lambda x: (x.split(">")[0],) + tuple(x.split(">")[1].split(":")))
        dataframe.sort_index(axis=1, inplace=True)
    else:
        comparison = [(targetval,)]
        dataframe.columns = dataframe.columns.map(lambda x: (targetval,) + tuple(x.split(":")))
    compareorder_unpacked = [tuple(x.split(",")) for x in compareorder]
    if len(compareorder_unpacked) > 0:
        comparison = sorted(
            comparison,
            key=lambda x: (
                compareorder_unpacked.index(x)
                if compareorder_unpacked is not None and x in compareorder_unpacked
                else len(comparison)
            ),
        )

    figure: Optional[Figure] = None
    if axes is None:
        figure = plt.figure(figsize=(10, 8))
        subplots = figure.subplots()
        assert isinstance(subplots, plt.Axes)
        axes = subplots
    else:
        figure = axes.get_figure()
    assert figure is not None

    labels: List[str] = []
    for i, comp in enumerate(comparison):
        formatdict = dict(zip(compare, comp))
        if summary is not None:
            comp_summary = summary
            for c in comp:
                comp_summary = comp_summary.get(c, comp_summary)
            formatdict.update(comp_summary)
        label = labelformat % formatdict
        label = replace_keywords(label, keyword_replacements)
        labels.append(label)

    centercol = VALUE_MEASURE_C[aggmode]
    split_colors = dict((tuple(k.split(",")), v) for (k, v) in colors.items()) if colors is not None else None
    comp_colors = get_colors(comparison, colors=split_colors)
    texts = []
    ymin, ymax = np.inf, -np.inf
    for i, comp in enumerate(comparison):
        if ",".join(str(c) for c in comp) in dontcompare:
            continue
        cols: List[str] = dataframe[comp][targetval].columns.to_list()
        uppercol = next(c for c in cols if c.endswith("-h"))
        lowercol = next(c for c in cols if c.endswith("-l"))
        upper = dataframe[comp][targetval][uppercol].to_numpy()
        lower = dataframe[comp][targetval][lowercol].to_numpy()
        ymax, ymin = max(ymax, np.max(upper)), min(ymin, np.min(lower))
        if errdisplay == ErrorDisplay.SHADE:
            axes.fill_between(dataframe.index.values, upper, lower, color=comp_colors[i], alpha=0.2)
        elif errdisplay == ErrorDisplay.BAR:
            center = dataframe[comp][targetval][centercol]
            xval = dataframe.index.values
            yval = center.to_numpy()
            yerr = np.abs(np.stack([lower, upper]) - yval)
            axes.errorbar(
                xval,
                yval,
                yerr=yerr,
                fmt="o",
                linewidth=DEFAULT_LINEWIDTH,
                capsize=6,
                color=comp_colors[i],
                label=labels[i],
            )
            if annotations:
                for x, y in zip(xval, yval):
                    text = axes.annotate(
                        "%.2f" % y,
                        xy=(x, y),
                        xytext=(fontsize * 0.5, 0),
                        textcoords="offset points",
                        fontsize=fontsize,
                        horizontalalignment="left",
                        verticalalignment="center",
                        # arrowprops=dict(arrowstyle="-", color=comp_colors[i]),
                    )
                    texts.append(text)

    linedesc = list(zip(comparison, comp_colors, labels))
    for comp, c, l in linedesc:
        if ",".join(str(c) for c in comp) in dontcompare:
            continue
        ll = l if errdisplay != ErrorDisplay.BAR else None
        axes.plot(
            dataframe[comp][targetval][centercol],
            color=c,
            label=ll,
            marker=linemarker,
            markersize=4 * DEFAULT_LINEWIDTH,
            linewidth=DEFAULT_LINEWIDTH,
        )

    # Plot a dashed line over the current lines to improve visibility of overlapping lines.
    for comp, c, l in reversed(linedesc):
        if ",".join(str(c) for c in comp) in dontcompare:
            continue
        axes.plot(dataframe[comp][targetval][centercol], color=c, linestyle=(0, (3, 3)), linewidth=DEFAULT_LINEWIDTH)

    # axes.set_xlim([dataframe.index.values[0], (dataframe.index.values[-1] - dataframe.index.values[0]) * 1.2])
    # axes.set_ylim([-0.2, 1])
    axes.set_ylabel(replace_keywords(targetval, keyword_replacements), fontsize=fontsize, wrap=True)
    axes.set_xlabel(replace_keywords(index, keyword_replacements), fontsize=fontsize, wrap=True)
    # axes.legend(loc="lower right", fontsize=fontsize, borderaxespad=0, edgecolor="black", fancybox=False)

    axes.relim()

    if len(texts) > 0:
        figure.canvas.draw()
        fix_text_positioning(texts, axes)

    axes.autoscale_view(tight=True)

    # Readjust the y-axis limits to include the error bars.
    ymin = min(ymin - 0.1 * (ymax - ymin), axes.get_ylim()[0])
    ymax = max(ymax + 0.1 * (ymax - ymin), axes.get_ylim()[1])
    if axes.get_yscale() == "log":
        ymin = max(0.01 * ymax, ymin)
    axes.set_ylim(ymin, ymax)

    figure.canvas.draw()

    return figure


T = TypeVar("T")


def dictpivot(d: Dict, compare: List[str], prefix: Tuple[str, ...] = tuple()) -> Dict[Tuple[str, ...], Dict]:
    if len(compare) > 0:
        result: Dict[Tuple[str, ...], Dict] = {}
        for k, v in d.items():
            dd = dictpivot(v, compare=compare[1:], prefix=prefix + (k,))
            result.update(dd)
        return result
    elif len(prefix) > 0:
        return {prefix: d}
    else:
        return d

    # for k, v in d.items():
    #     if isinstance(v, dict):
    #         for item in dictpivot(v, prefix=prefix + (k,)):
    #             yield item
    #     else:
    #         yield prefix + (k,), v


def barplot(
    summary: dict,
    targetval: str,
    compare: Optional[List[str]] = None,
    compareorder: List[str] = [],
    colors: Optional[Dict[str, str]] = None,
    labelformat: Optional[str] = None,
    aggmode: AggregationMode = AggregationMode.MEDIAN_PERC_90,
    keyword_replacements: Optional[Dict[str, str]] = None,
    axes: Optional[plt.Axes] = None,
    fontsize: int = DEFAULT_FONTSIZE,
    annotations: bool = False,
    dontcompare: Optional[str] = None,
) -> Optional[Figure]:
    if compare is None:
        compare = []
    # if len(compare) == 0:
    #     compare = ["value"]
    #     summary = {"value": summary}

    if labelformat is None:
        if len(compare) > 0:
            labelformat = "; ".join("%%(%s)s" % c for c in compare)
        else:
            labelformat = targetval
    if keyword_replacements is None:
        keyword_replacements = {}
    if dontcompare is None:
        dontcompare = ""
    figure: Optional[Figure] = None
    if axes is None:
        figure = plt.figure(figsize=(10, 8))
        subplots = figure.subplots()
        assert isinstance(subplots, plt.Axes)
        axes = subplots
    else:
        figure = axes.get_figure()
    assert figure is not None

    # x, y, yerr = [], [], []
    summary = dictpivot(summary, compare=compare)
    summary_items = list(summary.items())

    compareorder_unpacked = [tuple(x.split(",")) for x in compareorder]
    if len(compareorder_unpacked) > 0:
        summary_items = sorted(
            summary_items,
            key=lambda x: (
                compareorder_unpacked.index(x[0])
                if compareorder_unpacked is not None and x[0] in compareorder_unpacked
                else len(summary_items)
            ),
        )

    comparison = [item[0] for item in summary_items]
    split_colors = dict((tuple(k.split(",")), v) for (k, v) in colors.items()) if colors is not None else None
    comp_colors = get_colors(comparison, colors=split_colors)
    texts = []
    ymin, ymax = np.inf, -np.inf

    for i, (comp, values) in enumerate(summary_items):
        if ",".join(str(c) for c in comp) in dontcompare:
            continue
        centercol = VALUE_MEASURE_C[aggmode]
        formatdict = dict(zip(compare, comp))
        label = labelformat % formatdict
        label = replace_keywords(label, keyword_replacements)
        # xval = "; ".join(str(x) for x in comp)
        col = targetval + ":" + centercol
        yval = values[col]

        uppercol = next(c for c in values.keys() if c.startswith(targetval) and c.endswith("-h"))
        lowercol = next(c for c in values.keys() if c.startswith(targetval) and c.endswith("-l"))
        yerr = np.abs(np.array([[values[col] - values[lowercol]], [values[col] - values[uppercol]]]))
        ymax, ymin = max(ymax, np.max(yerr)), min(ymin, np.min(yerr))
        axes.errorbar(
            [i],
            [yval],
            yerr=yerr,
            fmt="o",
            linewidth=DEFAULT_LINEWIDTH,
            capsize=6,
            color=comp_colors[i],
            label=label,
        )

        if annotations:
            text = axes.annotate(
                "%.2f" % yval,
                xy=(i, yval),
                xytext=(fontsize * 0.5, 0),
                textcoords="offset points",
                fontsize=fontsize,
                horizontalalignment="left",
                verticalalignment="center",
            )
            texts.append(text)

    axes.set_ylabel(replace_keywords(targetval, keyword_replacements), fontsize=fontsize, wrap=True)
    axes.get_xaxis().set_ticks([])
    axes.relim()

    # Squeeze xlimit.
    xlim = axes.get_xlim()
    xlim_delta = xlim[1] - xlim[0]
    axes.set_xlim((xlim[0] - xlim_delta * 0.1, xlim[1] + xlim_delta * 0.3))
    # axes.legend(
    #     loc="upper right", fontsize=fontsize, borderaxespad=0, edgecolor="black", fancybox=False, ncol=len(summary)
    # )

    axes.relim()

    if len(texts) > 0:
        figure.canvas.draw()
        fix_text_positioning(texts, axes)

    axes.autoscale_view(tight=True)
    figure.canvas.draw()

    # Readjust the y-axis limits to include the error bars.
    ymin = min(ymin - 0.1 * (ymax - ymin), axes.get_ylim()[0])
    ymax = max(ymax + 0.1 * (ymax - ymin), axes.get_ylim()[1])
    if axes.get_yscale() == "log":
        ymin = max(0.01 * ymax, ymin)
    axes.set_ylim(ymin, ymax)

    return figure


def dotplot(
    summary: dict,
    xtargetval: str,
    ytargetval: str,
    compare: Optional[List[str]] = None,
    compareorder: List[str] = [],
    colors: Optional[Dict[str, str]] = None,
    labelformat: Optional[str] = None,
    aggmode: AggregationMode = AggregationMode.MEDIAN_PERC_90,
    keyword_replacements: Optional[Dict[str, str]] = None,
    axes: Optional[plt.Axes] = None,
    fontsize: int = DEFAULT_FONTSIZE,
    annotations: bool = False,
    dontcompare: Optional[str] = None,
) -> Optional[Figure]:
    if compare is None:
        compare = []
    # if len(compare) == 0:
    #     compare = ["value"]
    #     summary = {"value": summary}

    if labelformat is None:
        if len(compare) > 0:
            labelformat = "; ".join("%%(%s)s" % c for c in compare)
        else:
            labelformat = "; ".join("%%(%s)s" % c for c in [xtargetval, ytargetval])
    if keyword_replacements is None:
        keyword_replacements = {}
    if dontcompare is None:
        dontcompare = ""
    figure: Optional[Figure] = None
    if axes is None:
        figure = plt.figure(figsize=(10, 8))
        subplots = figure.subplots()
        assert isinstance(subplots, plt.Axes)
        axes = subplots
    else:
        figure = axes.get_figure()
    assert figure is not None

    # x, y, yerr = [], [], []
    summary = dictpivot(summary, compare=compare)
    summary_items = list(summary.items())

    compareorder_unpacked = [tuple(x.split(",")) for x in compareorder]
    if len(compareorder_unpacked) > 0:
        summary_items = sorted(
            summary_items,
            key=lambda x: (
                compareorder_unpacked.index(x[0])
                if compareorder_unpacked is not None and x[0] in compareorder_unpacked
                else len(summary_items)
            ),
        )

    comparison = [item[0] for item in summary_items]
    split_colors = dict((tuple(k.split(",")), v) for (k, v) in colors.items()) if colors is not None else None
    comp_colors = get_colors(comparison, colors=split_colors)
    texts = []

    for i, (comp, values) in enumerate(summary_items):
        if ",".join(str(c) for c in comp) in dontcompare:
            continue
        centercol = VALUE_MEASURE_C[aggmode]
        formatdict = dict(zip(compare, comp))
        label = labelformat % formatdict
        label = replace_keywords(label, keyword_replacements)
        # xval = "; ".join(str(x) for x in comp)
        xcol = xtargetval + ":" + centercol
        xval = values[xcol]
        ycol = ytargetval + ":" + centercol
        yval = values[ycol]

        xuppercol = next(c for c in values.keys() if c.startswith(xtargetval) and c.endswith("-h"))
        xlowercol = next(c for c in values.keys() if c.startswith(xtargetval) and c.endswith("-l"))
        xerr = np.abs(np.array([[values[xcol] - values[xlowercol]], [values[xcol] - values[xuppercol]]]))

        yuppercol = next(c for c in values.keys() if c.startswith(ytargetval) and c.endswith("-h"))
        ylowercol = next(c for c in values.keys() if c.startswith(ytargetval) and c.endswith("-l"))
        yerr = np.abs(np.array([[values[ycol] - values[ylowercol]], [values[ycol] - values[yuppercol]]]))
        axes.errorbar(
            [xval],
            [yval],
            xerr=xerr,
            yerr=yerr,
            fmt="o",
            linewidth=DEFAULT_LINEWIDTH,
            capsize=6,
            color=comp_colors[i],
            label=label,
        )

        if annotations:
            text = axes.annotate(
                "%.2f" % yval,
                xy=(xval, yval),
                xytext=(fontsize * 0.5, 0),
                textcoords="offset points",
                fontsize=fontsize,
                horizontalalignment="left",
                verticalalignment="center",
            )
            texts.append(text)

    axes.set_ylabel(replace_keywords(ytargetval, keyword_replacements), fontsize=fontsize, wrap=True)
    axes.set_xlabel(replace_keywords(xtargetval, keyword_replacements), fontsize=fontsize, wrap=True)
    axes.relim()

    # Squeeze xlimit.
    # xlim = axes.get_xlim()
    # xlim_delta = xlim[1] - xlim[0]
    # axes.set_xlim((xlim[0] - xlim_delta * 0.1, xlim[1] + xlim_delta * 0.3))
    # axes.legend(
    #     loc="upper right", fontsize=fontsize, borderaxespad=0, edgecolor="black", fancybox=False, ncol=len(summary)
    # )

    axes.relim()

    if len(texts) > 0:
        figure.canvas.draw()
        fix_text_positioning(texts, axes)

    axes.autoscale_view(tight=True)
    figure.canvas.draw()

    return figure


NONE_SYMBOL = "-"
DEFAULT_PLOTSIZE = [10, 8]


def ensurelist(
    x: Optional[Union[List[T], T]],
    default: Optional[T] = None,
    length: Optional[int] = None,
) -> List[T]:
    result: List[T] = ([default] if default is not None else []) if x is None else x if isinstance(x, list) else [x]
    if length is not None and len(result) != length:
        if len(result) > 1 and length > 0:
            raise ValueError("Expected length to be %d but encountered %d." % (length, len(result)))
        else:
            result = [result[0] for _ in range(length)]
    return result
    # return [x if x != NONE_SYMBOL else None for x in result]


def parseplotspec(spec: Union[Tuple[PlotType, List[str]], str]) -> Tuple[PlotType, List[str]]:
    if isinstance(spec, str):
        splits = spec.split(":")
        if len(splits) != 2 and len(splits) != 3:
            raise ValueError("The plot specifications must be formatted as plottype:target[:target].")
        plottype, target = splits[0], splits[1:]
        return (PlotType(plottype), target)
    return spec


def unpackdict(target: Dict[str, Union[Dict, Any]], prefix: str = "") -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for k, v in target.items():
        if isinstance(v, dict):
            for kk, vv in unpackdict(v, prefix=str(k)).items():
                key = ".".join(([] if prefix == "" else [prefix]) + [kk])
                result[key] = vv
        else:
            key = ".".join(([] if prefix == "" else [prefix]) + [k])
            result[key] = v
    return result


class AggregatePlot(Report, id="aggplot"):
    def __init__(
        self,
        dataframe: DataFrame,
        partvals: Optional[Dict[str, Any]] = None,
        id: Optional[str] = None,
        study: Optional[Study] = None,
        keyword_replacements: Optional[Dict[str, str]] = None,
        plot: Optional[Union[List[str], str]] = None,
        index: Optional[str] = None,
        targetval: Optional[Union[List[str], str]] = None,
        compare: Optional[Union[List[str], str]] = None,
        compareorder: Optional[List[str]] = None,
        colors: Optional[List[str]] = None,
        summarize: Optional[Union[List[str], str]] = None,
        crossaggover: Optional[Union[List[str], str]] = None,
        sliceby: Optional[Union[List[str], str]] = None,
        sliceop: Optional[SliceOp] = None,
        resultfilter: Optional[str] = None,
        errdisplay: Optional[ErrorDisplay] = None,
        aggmode: Optional[AggregationMode] = None,
        summode: Optional[AggregationMode] = None,
        crossaggmode: Optional[CrossAggregationMode] = None,
        xlogscale: Optional[Union[List[bool], bool]] = None,
        ylogscale: Optional[Union[List[bool], bool]] = None,
        xtickfmt: Optional[Union[List[TickFormat], TickFormat]] = None,
        ytickfmt: Optional[Union[List[TickFormat], TickFormat]] = None,
        linemarker: Optional[Union[List[str], str]] = None,
        annotations: Optional[Union[List[bool], bool]] = None,
        dontcompare: Optional[Union[List[str], str]] = None,
        labelformat: Optional[str] = None,
        plotsize: Optional[List[int]] = None,
        fontsize: Optional[int] = None,
        usetex: Optional[bool] = True,
        legend: Optional[bool] = True,
        titleformat: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            dataframe=dataframe,
            id=id,
            study=study,
            partvals=partvals,
            keyword_replacements=keyword_replacements,
            **kwargs,
        )
        self._plot = ensurelist(plot, default=None)
        n_plots = len(self._plot)
        self._index = index
        self._targetval: List[str] = ensurelist(targetval, default=None)
        self._compare: List[str] = ensurelist(compare, default=None)
        self._compareorder: List[str] = compareorder if compareorder is not None else []
        self._colors: List[str] = colors if colors is not None else []
        self._summarize: List[str] = ensurelist(summarize, default=None)
        self._crossaggover: List[str] = ensurelist(crossaggover, default=None)
        self._sliceby: List[str] = ensurelist(sliceby, default=None)
        self._sliceop = sliceop if sliceop is not None else SliceOp.FIRST
        self._resultfilter: Optional[str] = resultfilter
        self._errdisplay = errdisplay if errdisplay is not None else ErrorDisplay.SHADE
        self._aggmode = aggmode if aggmode is not None else AggregationMode.MEDIAN_PERC_95
        self._summode = summode if summode is not None else AggregationMode.MEDIAN_PERC_95
        self._crossaggmode = crossaggmode if crossaggmode is not None else CrossAggregationMode.MEAN_ABSOLUTE_ERROR
        self._xlogscale: List[bool] = ensurelist(xlogscale, default=False, length=n_plots)
        self._ylogscale: List[bool] = ensurelist(ylogscale, default=False, length=n_plots)
        self._xtickfmt: List[TickFormat] = ensurelist(xtickfmt, default=TickFormat.DEFAULT, length=n_plots)
        self._ytickfmt: List[TickFormat] = ensurelist(ytickfmt, default=TickFormat.DEFAULT, length=n_plots)
        self._linemarker: List[str] = ensurelist(linemarker, default="", length=n_plots)
        self._annotations: List[bool] = ensurelist(annotations, default=False, length=n_plots)
        self._dontcompare: List[str] = ensurelist(dontcompare, default=NONE_SYMBOL, length=n_plots)

        self._labelformat = (
            labelformat
            if labelformat is not None
            else (
                "; ".join("%%(%s)s" % x if x is not None else "" for x in self._compare)
                if len(self._compare) > 0
                else "%%(target)s"
            )
        )

        self._plotsize = plotsize if plotsize is not None else DEFAULT_PLOTSIZE
        if len(self._plotsize) != 2:
            raise ValueError("The plotsize must have two parameters: width and height.")
        self._fontsize = fontsize if fontsize is not None else DEFAULT_FONTSIZE
        self._usetex = usetex if usetex is not None else True
        self._legend = legend if legend is not None else True
        if titleformat is None:
            titleformat = ["; ".join("%s=%%(%s)s" % (str(k).title(), str(k)) for k in self._partvals.keys())]
        self._titleformat = titleformat

        self._view: Optional[DataFrame] = None
        self._figure: Optional[Figure] = None
        self._summary: Optional[dict] = None

    @attribute
    def plot(self) -> List[str]:
        """The type of plot to draw. Multiple plots can be drawn on the same figure."""
        return self._plot

    @attribute
    def targetval(self) -> List[str]:
        """The result attribute to evaluate."""
        return self._targetval

    @attribute
    def compare(self) -> List[str]:
        """The attribute to compare by."""
        return self._compare

    @attribute
    def compareorder(self) -> List[str]:
        """If specified, a list of comparison values."""
        return self._compareorder

    @attribute
    def colors(self) -> List[str]:
        """Assignments of colors to comparison values. Each assignment has to be formatted as comparevalue:color."""
        return self._colors

    @attribute
    def index(self) -> Optional[str]:
        """The attribute to use as index."""
        return self._index

    @attribute
    def summarize(self) -> List[str]:
        """The attribute aggregate over the entire dataframe."""
        return self._summarize

    @attribute
    def crossaggover(self) -> List[str]:
        """The attributes that we use for joining the cross aggregation."""
        return self._crossaggover

    @attribute
    def sliceby(self) -> List[str]:
        """Slice along the index attribute by taking the first value in each slice plane.
        We use the given set of attributes to define a slice plane."""
        return self._sliceby

    @attribute
    def sliceop(self) -> SliceOp:
        """If we perform slicing on the dataframe, this specifies the aggregation operation to perform on each slice."""
        return self._sliceop

    @attribute
    def resultfilter(self) -> Optional[str]:
        """A Pandas query that will be used to filter the results of the study."""
        return self._resultfilter

    @attribute
    def aggmode(self) -> AggregationMode:
        """The mode of aggregation to apply."""
        return self._aggmode

    @attribute
    def summode(self) -> AggregationMode:
        """The mode of summarization to apply."""
        return self._summode

    @attribute
    def crossaggmode(self) -> CrossAggregationMode:
        """The mode of cross aggregation to apply."""
        return self._crossaggmode

    @attribute
    def errdisplay(self) -> ErrorDisplay:
        """The method of displaying error bars."""
        return self._errdisplay

    @attribute
    def xlogscale(self) -> List[bool]:
        """Whether to represent the x-axis in the logarithmic scale."""
        return self._xlogscale

    @attribute
    def ylogscale(self) -> List[bool]:
        """Whether to represent the y-axis in the logarithmic scale."""
        return self._ylogscale

    @attribute
    def xtickfmt(self) -> List[TickFormat]:
        """The tick formatting to use for the x-axis."""
        return self._xtickfmt

    @attribute
    def ytickfmt(self) -> List[TickFormat]:
        """The tick formatting to use for the y-axis."""
        return self._ytickfmt

    @attribute
    def linemarker(self) -> List[str]:
        """The type of marker to use when plotting lines."""
        return self._linemarker

    @attribute
    def annotations(self) -> List[bool]:
        """Whether to add annotations to all points displayed."""
        return self._annotations

    @attribute
    def dontcompare(self) -> List[str]:
        """Space-separated list of comma separated lists of values that should not be compared."""
        return self._dontcompare

    @attribute
    def labelformat(self) -> str:
        """
        The string used to format labels in figures. A wildcard such as %%(attrname)s of %%(attrname).2f can be used.
        It will be replaced with a value where 'attrname' is either comparison attribute or a summary attribute.
        """
        return self._labelformat

    @attribute
    def plotsize(self) -> List[int]:
        """The width and height of each plot."""
        return self._plotsize

    @attribute
    def fontsize(self) -> int:
        """The size of text used in figures."""
        return self._fontsize

    @attribute
    def usetex(self) -> bool:
        """Whether to use LaTeX fonts."""
        return self._usetex

    @attribute
    def legend(self) -> bool:
        """Whether to show the legend."""
        return self._legend

    @attribute
    def titleformat(self) -> List[str]:
        """The formatted text to use in the title."""
        return self._titleformat

    def generate(self) -> None:
        dataframe = self.dataframe
        if self.resultfilter is not None:
            dataframe = dataframe.query(self.resultfilter)

        keyword_replacements: Dict[str, str] = {**self.keyword_replacements}
        summarydict: Dict[str, str] = {}
        if self.study is not None:
            keyword_replacements.update(self.study.get_keyword_replacements())

        if self.index is not None and len(self.targetval) > 0:
            if self._view is None:
                agg_dataframe = dataframe

                # If slicing was specified, then we slice the dataframe first.
                if len(self.sliceby) > 0:
                    groupattrs = self.sliceby + self.compare + [self.index]
                    groupby = agg_dataframe.groupby(groupattrs)
                    sliceop = SLICE_OPS[self.sliceop]
                    agg_dataframe = sliceop(groupby)
                    agg_dataframe.reset_index(inplace=True)

                if len(self._crossaggover) > 0:
                    agg_dataframe = cross_aggregate(
                        dataframe=dataframe,
                        index=[self.index],
                        compare=self.compare,
                        targetval=self.targetval,
                        crossaggover=self._crossaggover,
                        crossaggpairs=self._compareorder,
                        crossaggmode=self._crossaggmode,
                    )

                self._view = aggregate(
                    dataframe=agg_dataframe,
                    compare=self.compare,
                    index=self.index,
                    targetval=self.targetval,
                    aggmode=self.aggmode,
                )

        if len(self.summarize) > 0:
            if self._summary is None:
                summ_dataframe = dataframe

                # If slicing was specified, then we slice the dataframe first.
                if len(self.sliceby) > 0:
                    groupattrs = self.sliceby + self.compare
                    groupby = summ_dataframe.groupby(groupattrs)
                    sliceop = SLICE_OPS[self.sliceop]
                    summ_dataframe = sliceop(groupby)
                    summ_dataframe.reset_index(inplace=True)

                if len(self._crossaggover) > 0:
                    summ_dataframe = cross_aggregate(
                        dataframe=summ_dataframe,
                        index=None,
                        compare=self.compare,
                        targetval=self.targetval,
                        crossaggover=self._crossaggover,
                        crossaggpairs=self._compareorder,
                        crossaggmode=self._crossaggmode,
                    )

                self._summary = summarize(
                    dataframe=summ_dataframe,
                    summarize=self.summarize,
                    compare=self.compare,
                    summode=self.summode,
                )
            summarydict = dict((k, represent(v)) for (k, v) in unpackdict(self._summary).items())

        if len(self.plot) > 0:
            figsize = (len(self.plot) * self.plotsize[0], self.plotsize[1])
            self._figure = plt.figure(figsize=figsize)
            formatdict = dict(**self.partvals)
            formatdict.update(summarydict)
            title = "\n".join(self.titleformat) % formatdict
            title = replace_keywords(title, keyword_replacements)
            plt.rc("text", usetex=self.usetex)
            if not title.isspace():
                self._figure.suptitle(title, fontsize=self.fontsize * 0.9)
            subplots = self._figure.subplots(nrows=1, ncols=len(self.plot))
            assert isinstance(subplots, np.ndarray) or isinstance(subplots, plt.Axes)
            axes: NDArray = np.array([subplots]) if isinstance(subplots, plt.Axes) else subplots.flatten()
            colors = dict((x.split(":")[0], x.split(":")[1]) for x in self._colors) if len(self._colors) > 0 else None

            for i, plotspec in enumerate(self.plot):
                plottype, target = parseplotspec(plotspec)

                # Axes style.
                axes[i].spines["top"].set_visible(False)
                axes[i].spines["right"].set_visible(False)
                axes[i].tick_params(axis="both", which="major", labelsize=self.fontsize)

                if self.xlogscale[i]:
                    axes[i].set_xscale("log")
                if self.ylogscale[i]:
                    axes[i].set_yscale("log")

                if self.xtickfmt[i] == TickFormat.PERCENT:
                    axes[i].xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
                elif self.xtickfmt[i] == TickFormat.ENG:
                    axes[i].xaxis.set_major_formatter(EngFormatter(places=0, sep=""))
                if self.ytickfmt[i] == TickFormat.PERCENT:
                    axes[i].yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
                elif self.ytickfmt[i] == TickFormat.ENG:
                    axes[i].yaxis.set_major_formatter(EngFormatter(places=0, sep=""))

                if plottype == PlotType.LINE:
                    if self._index is None:
                        raise ValueError("An index must be specified when plotting line plots.")
                    if self._view is None:
                        raise ValueError(
                            "An aggregate view must be generated for a line plot by specifying a targetval and index."
                        )

                    lineplot(
                        self._view,
                        index=self._index,
                        targetval=target[0],
                        summary=self._summary,
                        compare=self._compare,
                        compareorder=self._compareorder,
                        colors=colors,
                        errdisplay=self._errdisplay,
                        labelformat=self._labelformat,
                        aggmode=self._aggmode,
                        keyword_replacements=keyword_replacements,
                        axes=axes[i],
                        fontsize=self.fontsize,
                        annotations=self._annotations[i],
                        dontcompare=self._dontcompare[i],
                        linemarker=self._linemarker[i],
                    )
                elif plottype == PlotType.BAR:
                    if self._summary is None:
                        raise ValueError("A bar plot can only be generated from a summary.")

                    barplot(
                        summary=self._summary,
                        targetval=target[0],
                        compare=self.compare,
                        compareorder=self._compareorder,
                        colors=colors,
                        labelformat=self._labelformat,
                        aggmode=self._summode,
                        keyword_replacements=keyword_replacements,
                        axes=axes[i],
                        fontsize=self.fontsize,
                        annotations=self._annotations[i],
                        dontcompare=self._dontcompare[i],
                    )

                else:
                    if self._summary is None:
                        raise ValueError("A dot plot can only be generated from a summary.")

                    if len(target) != 2:
                        raise ValueError(
                            "A dot plot must be specified with two targets. One for the x-axis and one for the y-axis."
                        )

                    dotplot(
                        summary=self._summary,
                        xtargetval=target[0],
                        ytargetval=target[1],
                        compare=self.compare,
                        compareorder=self._compareorder,
                        colors=colors,
                        labelformat=self._labelformat,
                        aggmode=self._summode,
                        keyword_replacements=keyword_replacements,
                        axes=axes[i],
                        fontsize=self.fontsize,
                        annotations=self._annotations[i],
                        dontcompare=self._dontcompare[i],
                    )

            if self._legend:
                self._figure.subplots_adjust(bottom=0.25)
                lines, labels = self._figure.axes[0].get_legend_handles_labels()
                self._figure.legend(
                    lines,
                    labels,
                    loc="upper center",
                    bbox_to_anchor=(0.5, -0.05),
                    fontsize=self.fontsize,
                    # borderaxespad=0,
                    edgecolor="black",
                    fancybox=False,
                    ncol=len(lines),
                )
            self._figure.tight_layout()

    @result
    def view(self) -> Optional[DataFrame]:
        return self._view

    @result
    def figure(self) -> Optional[Figure]:
        return self._figure

    @result
    def summary(self) -> Optional[dict]:
        return self._summary
