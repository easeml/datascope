import logging
import os
import zerorpc

from tqdm import tqdm
from typing import Any, Optional, Sequence

from .scenarios import (
    Study,
    Scenario,
    Backend,
    Report,
    QueueProtocol,
    get_scenario_runner,
    DEFAULT_RESULTS_PATH,
    DEFAULT_RESULTS_SCENARIOS_PATH,
    DEFAULT_STUDY_PATH,
    DEFAULT_BACKEND,
    DEFAULT_SLURM_JOBMEMORY,
)


def run(
    output_path: str = DEFAULT_RESULTS_PATH,
    backend: Backend = DEFAULT_BACKEND,
    no_save: bool = False,
    ray_address: Optional[str] = None,
    ray_numprocs: Optional[int] = None,
    slurm_jobmemory: Optional[str] = DEFAULT_SLURM_JOBMEMORY,
    **attributes: Any
) -> None:
    # print("run", output_path, attributes)

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
        study.save()

    # Run the study.
    study.run(
        backend=backend,
        ray_address=ray_address,
        ray_numprocs=ray_numprocs,
        slurm_jobmemory=slurm_jobmemory,
        eagersave=not no_save,
    )

    # Save the study.
    if not no_save:
        study.save()


def run_scenario(
    output_path: str = DEFAULT_RESULTS_SCENARIOS_PATH,
    no_save: bool = False,
    event_server: Optional[str] = None,
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
            raise ValueError("The provided attributes correspond to more than one valid scenarios.")
        scenario = scenarios[0]
        path = os.path.join(output_path, scenario.id)

        # Eagerly save the scenario.
        if not no_save:
            scenario.save(path)
    queue: Optional[QueueProtocol] = None
    pickled_queue = False
    if event_server is not None:
        client = zerorpc.Client()
        client.connect(event_server)
        queue = client
        pickled_queue = True
    runner = get_scenario_runner(queue=queue, pickled_queue=pickled_queue)

    scenario.logger.setLevel(logging.DEBUG)
    scenario = runner(scenario)

    # Save the scenario.
    if not no_save:
        scenario.save(path)


def report(
    study_path: Optional[str] = DEFAULT_STUDY_PATH,
    groupby: Optional[Sequence[str]] = None,
    output_path: Optional[str] = None,
    saveonly: Optional[Sequence[str]] = None,
    use_subdirs: bool = False,
    **attributes: Any
) -> None:

    if study_path is None:
        raise ValueError("The provided study path cannot be None.")

    # Load the study.
    study = Study.load(study_path)

    # Get applicable instances of reports.
    reports = list(Report.get_instances(study=study, groupby=groupby, **attributes))

    for r in tqdm(reports, desc="Reports"):
        r.generate()
        r.save(path=output_path, use_subdirs=use_subdirs, saveonly=saveonly)
