import argparse
import importlib
import os
import sys

from typing import Any, Optional

from .commands import run, run_scenario, report, cache_pipelines
from ..datasets import preload_datasets, DEFAULT_BATCH_SIZE, DEFAULT_CACHE_DIR, Dataset
from ..pipelines import Pipeline
from .base import (
    Scenario,
    Report,
    Backend,
    DEFAULT_RESULTS_PATH,
    DEFAULT_RESULTS_SCENARIOS_PATH,
    DEFAULT_STUDY_PATH,
    DEFAULT_BACKEND,
    DEFAULT_SLURM_JOBTIME,
    DEFAULT_SLURM_JOBMEMORY,
)

sys.path.append(os.getcwd())
MODULES_INCLUDE_VARNAME = "DATASCOPE_EXPERIMENTS_MODULES_INCLUDE"
MODULES_INCLUDE = (
    [importlib.import_module(m.strip()) for m in os.environ[MODULES_INCLUDE_VARNAME].split(",")]
    if MODULES_INCLUDE_VARNAME in os.environ
    else []
)


# Courtesy of http://stackoverflow.com/a/10551190 with env-var retrieval fixed
class EnvDefault(argparse.Action):
    """An argparse action class that auto-sets missing default values from env
    vars. Defaults to requiring the argument."""

    def __init__(self, envvar: Optional[str], required: bool = True, default: Optional[Any] = None, **kwargs):
        if default is not None and envvar is not None:
            if envvar in os.environ:
                default = os.environ[envvar]
        if required and default is not None:
            required = False
        super(EnvDefault, self).__init__(default=default, required=required, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)


def env_default(envvar: Optional[str]):
    def wrapper(**kwargs):
        return EnvDefault(envvar, **kwargs)

    return wrapper


def main():
    parser = argparse.ArgumentParser(
        prog="experiments", description="This command allows interaction with datascope experiments."
    )
    subparsers = parser.add_subparsers(title="commands", help="Available commands.", dest="command")

    parser_run = subparsers.add_parser("run")

    parser_run.add_argument(
        "-o",
        "--output-path",
        type=str,
        action=env_default("EXPERIMENTS_RESULTS_PATH"),
        default=DEFAULT_RESULTS_PATH,
        help="Path where the data is stored. Default: '%s'" % DEFAULT_RESULTS_PATH,
    )

    parser_run.add_argument(
        "-b",
        "--backend",
        type=Backend,
        choices=[x.value for x in list(Backend)],
        action=env_default("EXPERIMENTS_BACKEND"),
        default=DEFAULT_BACKEND,
        help="The backend to run against. Default: '%s'" % str(DEFAULT_BACKEND.value),
    )

    parser_run.add_argument(
        "--no-save",
        action="store_true",
        help="Prevent saving the study.",
    )

    parser_run.add_argument(
        "--ray-address",
        type=str,
        action=env_default("EXPERIMENTS_RAY_ADDRESS"),
        required=False,
        default=None,
        help="Address of the ray server. If omitted, a new server ad-hoc will be created.",
    )

    parser_run.add_argument(
        "--ray-numprocs",
        type=int,
        action=env_default("EXPERIMENTS_RAY_NUMPROCS"),
        required=False,
        default=None,
        help="Number of ray processes to start if running in parallel. Defaults to the number of cores.",
    )

    parser_run.add_argument(
        "--slurm-jobtime",
        type=str,
        action=env_default("EXPERIMENTS_SLURM_JOBTIME"),
        default=DEFAULT_SLURM_JOBTIME,
        help="Runtime requirement for each slurm job (if slurm is used as backend).",
    )

    parser_run.add_argument(
        "--slurm-jobmemory",
        type=str,
        action=env_default("EXPERIMENTS_SLURM_JOBMEMORY"),
        default=DEFAULT_SLURM_JOBMEMORY,
        help="Memory requirements for each slurm job (if slurm is used as backend).",
    )

    parser_run.add_argument(
        "--slurm-constraint",
        type=str,
        action=env_default("EXPERIMENTS_SLURM_CONSTRAINT"),
        required=False,
        default=None,
        help="Constraint to be specified for selecting slurm cluster nodes (if slurm is used as backend).",
    )

    parser_run.add_argument(
        "--slurm-partition",
        type=str,
        action=env_default("EXPERIMENTS_SLURM_PARTITION"),
        required=False,
        default=None,
        help="The slurm partition to target (if slurm is used as backend).",
    )

    parser_run.add_argument(
        "--slurm-maxjobs",
        type=int,
        action=env_default("EXPERIMENTS_SLURM_MAXJOBS"),
        required=False,
        default=None,
        help="Maimum number of slurm jobs that will be running at any given time (if slurm is used as backend).",
    )

    parser_run.add_argument(
        "--slurm-args",
        type=str,
        action=env_default("EXPERIMENTS_SLURM_ARGS"),
        required=False,
        default=None,
        help="Additional arguments to pass to sbatch (if slurm is used as backend).",
    )

    parser_run.add_argument(
        "--eventstream-host-ip",
        type=str,
        action=env_default("EVENTSTREAM_HOST_IP"),
        required=False,
        default=None,
        help="The IP address to use for receiving events from distributed jobs (if slurm is used as backend).",
    )

    parser_run.add_argument(
        "--eventstream-host-port",
        type=int,
        action=env_default("EVENTSTREAM_HOST_PORT"),
        required=False,
        default=None,
        help="The port to use for receiving events from distributed jobs (if slurm is used as backend).",
    )

    # Build arguments from scenario attributes.
    Scenario.add_dynamic_arguments(parser=parser_run, all_iterable=True, single_instance=False)

    parser_run_scenario = subparsers.add_parser("run-scenario")

    parser_run_scenario.add_argument(
        "-o",
        "--output-path",
        type=str,
        default=DEFAULT_RESULTS_SCENARIOS_PATH,
        help="Path where the data is stored. Default: '%s'" % DEFAULT_RESULTS_SCENARIOS_PATH,
    )

    parser_run_scenario.add_argument(
        "--no-save",
        action="store_true",
        help="Prevent saving the scenario.",
    )

    parser_run_scenario.add_argument(
        "-e",
        "--event-server",
        type=str,
        default=None,
        help="Address of the event server. If specified, logging and progress events will be streamed to it.",
    )

    parser_run_scenario.add_argument(
        "-m",
        "--job-memory",
        type=str,
        default=None,
        help="The amount of memory allowed for a job.",
    )

    # Build arguments from scenario attributes.
    Scenario.add_dynamic_arguments(parser=parser_run_scenario, all_iterable=True, single_instance=True)

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
        "-p",
        "--partby",
        type=str,
        nargs="+",
        help="List of columns used to partition results. Each partition will result in a single report instance.",
    )

    parser_report.add_argument(
        "--saveonly",
        type=str,
        nargs="+",
        help="List of result attributes to save.",
    )

    parser_report.add_argument(
        "--use-subdirs",
        action="store_true",
        help="Store report artifacts in a hierarchy of subdirectories based on partition keys and values.",
    )

    parser_report.add_argument(
        "--no-multiprocessing",
        action="store_true",
        help="Do not use multiprocessing for generating reports. Run in single threaded mode.",
    )

    # Build arguments from report attributes.
    Report.add_dynamic_arguments(parser=parser_report, all_iterable=False, single_instance=False)

    subparsers.add_parser("preload-datasets")

    parser_cache_pipeline = subparsers.add_parser("cache-pipelines")

    parser_cache_pipeline.add_argument(
        "-d",
        "--datasets",
        nargs="+",
        type=str,
        choices=Dataset.get_subclasses().keys(),
        help="The target dataset for which to construct the pipeline cache.",
    )

    parser_cache_pipeline.add_argument(
        "-p",
        "--pipelines",
        nargs="+",
        type=str,
        choices=Pipeline.get_subclasses().keys(),
        help="The pipeline to run over the target dataset.",
    )

    parser_cache_pipeline.add_argument(
        "--cache-dir",
        type=str,
        default=DEFAULT_CACHE_DIR,
        help="The base directory in which pipeline output data will be stored.",
    )

    parser_cache_pipeline.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="The maximum size of a single batch of data that will be passed through the pipeline.",
    )
    args = parser.parse_args()
    kwargs = vars(args)

    if args.command == "run":
        run(**kwargs)
    elif args.command == "run-scenario":
        run_scenario(**kwargs)
    elif args.command == "report":
        report(**kwargs)
    elif args.command == "preload-datasets":
        preload_datasets(**kwargs)
    elif args.command == "cache-pipelines":
        cache_pipelines(**kwargs)
    else:
        parser.print_help()
