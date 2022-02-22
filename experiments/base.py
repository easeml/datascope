import os

from experiments.scenarios.base import Report
from tqdm import tqdm
from typing import Any, Optional, Sequence

from .scenarios import Study, Scenario, DEFAULT_RESULTS_PATH, DEFAULT_STUDY_PATH


def run(
    output_path: str = DEFAULT_RESULTS_PATH,
    no_parallelism: bool = False,
    no_save: bool = False,
    ray_address: Optional[str] = None,
    ray_numprocs: Optional[int] = None,
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
    study = Study(scenarios=scenarios, outpath=output_path)

    # Run the study.
    study.run(parallel=not no_parallelism, ray_address=ray_address, ray_numprocs=ray_numprocs, eagersave=not no_save)

    # Save the study.
    if not no_save:
        study.save()


def finalize(
    study_path: Optional[str] = DEFAULT_STUDY_PATH,
    groupby: Optional[Sequence[str]] = None,
    output_path: Optional[str] = None,
    **attributes: Any
) -> None:

    if study_path is None:
        raise ValueError("The provided study path cannot be None.")

    # Load the study.
    study = Study.load(study_path)

    # Get applicable instances of reports.
    reports = list(Report.get_instances(study=study, groupby=groupby, **attributes))

    for report in tqdm(reports, desc="Reports"):
        report.generate()
        report.save(path=output_path)
