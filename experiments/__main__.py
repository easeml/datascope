import argparse
from enum import Enum
from typing import Any, Callable, List, Optional

from .base import run, finalize, DEFAULT_OUTPUT_PATH
from .scenarios import Scenario


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
        default=DEFAULT_OUTPUT_PATH,
        help="Path where the data is stored. Default: '%s'" % DEFAULT_OUTPUT_PATH,
    )

    parser_run.add_argument(
        "--no-parallelism",
        action="store_true",
        help="Prevent parallel execution of scenarios.",
    )

    parser_run.add_argument(
        "--ray-address",
        type=str,
        default=None,
        help="Address of the ray server. If omitted, a new server ad-hoc will be created.",
    )

    # Build arguments from scenario attributes.
    for name in Scenario.attribute_domains:
        dtype = Scenario.attribute_types[name]
        default = Scenario.attribute_defaults[name]
        domain: Optional[List] = [x.value if isinstance(x, Enum) else x for x in Scenario.attribute_domains[name]]
        if domain == [None]:
            domain = None
        helpstring = Scenario.attribute_helpstrings[name] or ("Scenario " + name + ".")
        if default is None:
            helpstring += " Default: [all]"
        else:
            helpstring += " Default: %s" % str(default)
        parser_run.add_argument(
            "--%s" % name,
            help=helpstring,
            type=make_type_parser(Scenario.attribute_types[name]),
            choices=domain,
            nargs="+",
        )

    parser_finalize = subparsers.add_parser("finalize")

    args = parser.parse_args()

    if args.command == "run":
        kwargs = vars(args)
        run(**kwargs)
    elif args.command == "finalize":
        finalize()
    else:
        parser.print_help()
