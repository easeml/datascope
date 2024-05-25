import logging
import os
import traceback
import zerorpc

# from ray.util.multiprocessing import Pool
from glob import glob
from itertools import repeat
from multiprocessing import Pool, Lock
from multiprocessing.synchronize import Lock as LockType
from tqdm import tqdm
from typing import Any, Optional, Sequence, Tuple, List, Type

from ..datasets import DEFAULT_BATCH_SIZE, DEFAULT_CACHE_DIR, Dataset
from ..pipelines import Pipeline

from .base import (
    Study,
    Scenario,
    Backend,
    Report,
    Queue as QueueProtocol,
    get_scenario_runner,
    DEFAULT_RESULTS_PATH,
    DEFAULT_RESULTS_SCENARIOS_PATH,
    DEFAULT_STUDY_PATH,
    DEFAULT_BACKEND,
    DEFAULT_SLURM_JOBTIME,
    DEFAULT_SLURM_JOBMEMORY,
)


def run(
    output_path: str = DEFAULT_RESULTS_PATH,
    backend: Backend = DEFAULT_BACKEND,
    no_save: bool = False,
    ray_address: Optional[str] = None,
    ray_numprocs: Optional[int] = None,
    slurm_jobtime: Optional[str] = DEFAULT_SLURM_JOBTIME,
    slurm_jobmemory: Optional[str] = DEFAULT_SLURM_JOBMEMORY,
    slurm_constraint: Optional[str] = None,
    slurm_partition: Optional[str] = None,
    slurm_maxjobs: Optional[int] = None,
    slurm_args: Optional[str] = None,
    eventstream_host_ip: Optional[str] = None,
    eventstream_host_port: Optional[int] = None,
    **attributes: Any
) -> None:
    # If we should continue the execution of an existing study, then we should load it.
    study: Optional[Study] = None
    path = output_path
    if Study.isstudy(output_path):
        study = Study.load(output_path)
        path = os.path.dirname(output_path)

        # Apply the attribute filters to the scenario if any attributes were specified.
        if len(attributes) > 0:
            scenarios = study.get_scenarios(**attributes)
            study = Study(
                scenarios,
                study.id,
                outpath=path,
                scenario_path_format=study.scenario_path_format,
                logstream=study._logstream,
            )

    # Construct a study from a set of scenarios.
    scenarios = list(Scenario.get_instances(**attributes))
    if study is not None:
        existing_scenarios = list(study.scenarios)
        for cs in scenarios:
            if all(not s.is_match(cs) for s in study.scenarios):
                existing_scenarios.append(cs)
        scenarios = existing_scenarios
        study = Study(
            scenarios=scenarios,
            id=study.id,
            outpath=path,
            scenario_path_format=study.scenario_path_format,
            logstream=study._logstream,
        )
    else:
        study = Study(scenarios=scenarios, outpath=output_path)

    # Eagerly save the study with all unfinished scenarios.
    if not no_save:
        study.save(include_results=False)

    # Run the study.
    study.run(
        backend=backend,
        ray_address=ray_address,
        ray_numprocs=ray_numprocs,
        slurm_jobtime=slurm_jobtime,
        slurm_jobmemory=slurm_jobmemory,
        slurm_constraint=slurm_constraint,
        slurm_partition=slurm_partition,
        slurm_maxjobs=slurm_maxjobs,
        slurm_args=slurm_args,
        host_ip=eventstream_host_ip,
        host_port=eventstream_host_port,
        eagersave=not no_save,
    )

    # Save the study.
    if not no_save:
        study.save()


def run_scenario(
    output_path: str = DEFAULT_RESULTS_SCENARIOS_PATH,
    no_save: bool = False,
    event_server: Optional[str] = None,
    job_memory: Optional[str] = None,
    **attributes: Any
) -> None:
    # If we should continue the execution of an existing scenario, then we should load it.
    scenario: Optional[Scenario] = None
    path = output_path
    if Scenario.isscenario(path):
        scenario = Scenario.load(path=path)

    else:
        # Otherwise, we will try to use the provided attributes to instantiate a scenario.
        scenarios = list(Scenario.get_instances(**attributes))

        # print(len(scenarios))
        # for s in scenarios:
        #     print(s)

        if len(scenarios) == 0:
            raise ValueError("The provided attributes do not correspond to any valid scenario.")
        elif len(scenarios) > 1:
            raise ValueError("The provided attributes correspond to more than one valid scenario.")
        scenario = scenarios[0]
        path = os.path.join(output_path, scenario.id)

        # Eagerly save the scenario (config attributes only).
        if not no_save:
            scenario.save(path, include_results=False)
    queue: Optional[QueueProtocol] = None
    pickled_queue = False
    if event_server is not None:
        client = zerorpc.Client()
        client.connect(event_server)
        queue = client
        pickled_queue = True
    runner = get_scenario_runner(queue=queue, pickled_queue=pickled_queue, job_memory=job_memory)

    scenario.logger.setLevel(logging.DEBUG)
    scenario = runner(scenario)

    # Save the scenario.
    if not no_save:
        scenario.save(path)


def _report_generator(args: Tuple[Report, Optional[str], bool, Optional[Sequence[str]]]) -> str:
    report, output_path, use_subdirs, saveonly = args
    result = "Report(%s)" % ", ".join("%s=%s" % (k, str(v)) for (k, v) in report.partvals.items())
    try:
        report.generate()
        if _generator_lock is not None:
            _generator_lock.acquire()
        report.save(path=output_path, use_subdirs=use_subdirs, saveonly=saveonly)
        result += ": Generated and saved."
    except Exception:
        result += ": Failed." + "\n"
        result += traceback.format_exc()
    finally:
        if _generator_lock is not None:
            _generator_lock.release()
    return result


_generator_lock: Optional[LockType] = None


def _init_pool(lock: LockType):
    global _generator_lock
    _generator_lock = lock


def report(
    study_path: Optional[str] = DEFAULT_STUDY_PATH,
    partby: Optional[Sequence[str]] = None,
    output_path: Optional[str] = None,
    saveonly: Optional[Sequence[str]] = None,
    use_subdirs: bool = False,
    no_multiprocessing: bool = False,
    **attributes: Any
) -> None:
    if study_path is None:
        raise ValueError("The provided study path cannot be None.")

    study: Optional[Study] = None
    if Study.isstudy(study_path):
        # The provided path points to a study, load it.
        study = Study.load(study_path)
        print("Loaded study: %s" % study_path)
    else:
        all_subdirs = glob(os.path.join(study_path, "*"))
        if len(all_subdirs) > 0:
            sorted_subdirs = sorted(all_subdirs, key=lambda x: os.path.getmtime(x), reverse=True)
            for subdir in sorted_subdirs:
                if Study.isstudy(subdir):
                    study = Study.load(subdir)
                    print("Loaded study: %s" % subdir)
                    break
    if study is None:
        raise ValueError("Cannot find any study under the provided study path.")

    # Get applicable instances of reports.
    reports = list(Report.get_instances(study=study, partby=partby, **attributes))

    item_args = zip(reports, repeat(output_path), repeat(use_subdirs), repeat(saveonly))
    if no_multiprocessing:
        for result in tqdm((_report_generator(args) for args in item_args), desc="Reports", total=len(reports)):
            tqdm.write(result)
    else:
        lock = None if no_multiprocessing else Lock()
        with Pool(processes=None, initializer=_init_pool, initargs=(lock,)) as pool:
            for result in tqdm(pool.imap_unordered(_report_generator, item_args), desc="Reports", total=len(reports)):
                tqdm.write(result)


def cache_pipelines(
    datasets: List[str],
    pipelines: List[str],
    cache_dir: str = DEFAULT_CACHE_DIR,
    batch_size: int = DEFAULT_BATCH_SIZE,
    **kwargs
) -> None:
    for dataset_id in datasets:
        for pipeline_id in pipelines:
            cls_dataset: Type[Dataset] = Dataset.get_subclasses()[dataset_id]
            inst_dataset = cls_dataset()
            cls_pipeline = Pipeline.get_subclasses()[pipeline_id]
            inst_pipeline = cls_pipeline()
            pipeline = inst_pipeline.construct(inst_dataset)
            print("Processing dataset '%s' with pipeline '%s'." % (dataset_id, pipeline_id))
            cls_dataset.construct_apply_cache(pipeline=pipeline, cache_dir=cache_dir, batch_size=batch_size)
