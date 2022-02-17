import argparse
import experiments.reports
import experiments.scenarios

from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

from .base import run, finalize
from .scenarios import DEFAULT_RESULTS_PATH, DEFAULT_STUDY_PATH


def make_type_parser(target: Optional[type]) -> Callable[[str], Any]:
    def parser(source: str) -> Any:
        if target is None:
            return source
        result: Any = source
        if issubclass(target, int):
            result = int(result)
        if issubclass(target, float):
            result = float(result)
        if issubclass(target, Enum):
            result = target(result)
        return result

    return parser


def add_dynamic_arguments(
    parser: argparse.ArgumentParser,
    attribute_domains: Dict[str, Set],
    attribute_types: Dict[str, Optional[type]],
    attribute_defaults: Dict[str, Optional[Any]],
    attribute_helpstrings: Dict[str, Optional[str]],
    attribute_isiterable: Dict[str, bool],
) -> None:
    for name in attribute_domains:
        default = attribute_defaults[name]
        domain: Optional[List] = [x.value if isinstance(x, Enum) else x for x in attribute_domains[name]]
        if domain == [None]:
            domain = None
        helpstring = attribute_helpstrings[name] or ("Scenario " + name + ".")
        if default is None:
            helpstring += " Default: [all]"
        else:
            helpstring += " Default: %s" % str(default)
        parser.add_argument(
            "--%s" % name,
            help=helpstring,
            type=make_type_parser(attribute_types[name]),
            choices=domain,
            nargs="+" if attribute_isiterable[name] else None,  # type: ignore
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="experiments", description="This command allows interaction with datascope experiments."
    )
    subparsers = parser.add_subparsers(title="commands", help="Available commands.", dest="command")

    parser_run = subparsers.add_parser("run")

    parser_run.add_argument(
        "-o",
        "--output-path",
        type=str,
        default=DEFAULT_RESULTS_PATH,
        help="Path where the data is stored. Default: '%s'" % DEFAULT_RESULTS_PATH,
    )

    parser_run.add_argument(
        "--no-parallelism",
        action="store_true",
        help="Prevent parallel execution of scenarios.",
    )

    parser_run.add_argument(
        "--no-save",
        action="store_true",
        help="Prevent saving the scenario.",
    )

    parser_run.add_argument(
        "--ray-address",
        type=str,
        default=None,
        help="Address of the ray server. If omitted, a new server ad-hoc will be created.",
    )

    parser_run.add_argument(
        "--ray-numprocs",
        type=int,
        default=None,
        help="Number of ray processes to start if running in parallel. Defaults to the number of cores.",
    )

    # Build arguments from scenario attributes.
    attribute_isiterable = dict((k, True) for k in experiments.scenarios.Scenario.attribute_types.keys())
    add_dynamic_arguments(
        parser_run,
        experiments.scenarios.Scenario.attribute_domains,
        experiments.scenarios.Scenario.attribute_types,
        experiments.scenarios.Scenario.attribute_defaults,
        experiments.scenarios.Scenario.attribute_helpstrings,
        experiments.scenarios.Scenario.attribute_isiterable,
    )

    parser_report = subparsers.add_parser("report")

    parser_report.add_argument(
        "-s",
        "--study-path",
        type=str,
        default=DEFAULT_STUDY_PATH,
        help="Path where the target study is stored. Default: '%s'" % DEFAULT_STUDY_PATH,
    )

    parser_report.add_argument(
        "-o",
        "--output-path",
        type=str,
        help="Path where the reports are stored. Default: same as study path",
    )

    parser_report.add_argument(
        "-g",
        "--groupby",
        type=str,
        nargs="+",
        help="List of columns used to group results.",
    )

    # Build arguments from report attributes.
    add_dynamic_arguments(
        parser_report,
        experiments.scenarios.Report.attribute_domains,
        experiments.scenarios.Report.attribute_types,
        experiments.scenarios.Report.attribute_defaults,
        experiments.scenarios.Report.attribute_helpstrings,
        experiments.scenarios.Report.attribute_isiterable,
    )

    # print(Report.attribute_domains)

    args = parser.parse_args()
    kwargs = vars(args)

    if args.command == "run":
        run(**kwargs)
    elif args.command == "report":
        finalize(**kwargs)
    else:
        parser.print_help()
