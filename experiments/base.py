import os
from typing import Any, Optional

from .scenarios import Study, Scenario, DEFAULT_OUTPUT_PATH


def run(
    output_path: str = DEFAULT_OUTPUT_PATH,
    no_parallelism: bool = False,
    ray_address: Optional[str] = None,
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

    else:

        # Construct a study from a set of scenarios.
        scenarios = list(Scenario.get_instances(**attributes))
        study = Study(scenarios=scenarios, outpath=output_path)

    # Run the study.
    study.run(parallel=not no_parallelism, ray_address=ray_address, eagersave=True)

    # Save the study.
    study.save()


def finalize() -> None:
    pass
