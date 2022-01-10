import numpy as np
import os
import pandas as pd
import random
import re
import string
import traceback
import warnings
import yaml

from abc import ABC, abstractmethod, abstractproperty
from datetime import datetime
from enum import Enum
from glob import glob
from inspect import signature
from itertools import product
from numpy import ndarray
from pandas import DataFrame
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Type, get_origin, get_args


class PropertyTag(property):
    @classmethod
    def get_properties(cls: Type[property], target: object) -> Dict[str, property]:
        res: Dict[str, property] = {}
        for name in dir(target):
            if hasattr(target, name):
                member = getattr(target, name)
                if isinstance(member, cls):
                    res[name] = member
        return res


class attribute(PropertyTag):
    __isattribute__ = True


class result(PropertyTag):
    __isresult__ = True


def extract_simpletype(target: object) -> type:
    pass


def extract_enumtype(target: object) -> Optional[Type[Enum]]:
    if isinstance(target, type) and issubclass(target, Enum):
        return target
    else:
        origin = get_origin(target)
        if origin is not None:
            for arg in get_args(target):
                argtype = extract_enumtype(arg)
                if isinstance(argtype, type) and issubclass(argtype, Enum):
                    return argtype
        return None


def get_property_type(target: object, name: str) -> Optional[Type]:
    member = getattr(target, name)
    sign = signature(member)
    return sign.return_annotation if isinstance(sign.return_annotation, type) else None


def get_property_domain(target: object, name: str) -> List[Any]:
    member = getattr(target, name)
    if not isinstance(member, property):
        raise ValueError("The specified member '%s' is not a property." % name)
    if member.fget is None:
        raise ValueError("The specified member '%s' does not have a getter." % name)
    sign = signature(member.fget)
    enum = extract_enumtype(sign.return_annotation)
    if enum is None:
        return [None]
    else:
        return list(enum.__members__.values())


def get_property_value(target: object, name: str) -> Any:
    member = getattr(target, name)
    if isinstance(member, property) and member.fget is not None:
        return member.fget(target)
    else:
        return None


def has_attribute_value(target: object, name: str, value: Any, ignore_none: bool = True) -> bool:
    target_value = get_property_value(target, name)
    if ignore_none:
        return target_value is None or target_value == value
    else:
        return target_value == value


def save_dict(source: Dict[str, Any], dirpath: str, basename: str) -> None:
    basedict: Dict[str, Any] = dict((k, v) for (k, v) in source.items() if type(v) in [int, float, bool, str])
    if len(basedict) > 0:
        with open(os.path.join(dirpath, ".".join([basename, "yaml"])), "w") as f:
            yaml.safe_dump(basedict, f)

    for name, data in source.items():
        if name not in basedict:
            if isinstance(data, ndarray):
                filename = os.path.join(dirpath, ".".join([basename, name, "npy"]))
                np.save(filename, data)
            elif isinstance(data, DataFrame):
                filename = os.path.join(dirpath, ".".join([basename, name, "csv"]))
                data.to_csv(filename)
            elif isinstance(data, dict):
                filename = os.path.join(dirpath, ".".join([basename, name, "yaml"]))
                with open(filename) as f:
                    yaml.safe_dump(data, f)
            else:
                raise ValueError("Key '%s' has unsupported type '%s'." % (name, str(type(data))))


def load_dict(dirpath: str, basename: str) -> Dict[str, Any]:
    if not os.path.isdir(dirpath):
        raise ValueError("The provided path '%s' does not point to a directory." % dirpath)

    res: Dict[str, Any] = {}

    filename = os.path.join(dirpath, ".".join([basename, "yaml"]))
    if os.path.isfile(filename):
        with open(filename, "r") as f:
            res.update(yaml.safe_load(f))

    for path in glob(os.path.join(dirpath, basename) + "*"):
        filename = os.path.basename(path)
        base, ext = os.path.splitext(filename)
        name = base[len(basename) + 1 :]  # noqa: E203

        if ext == "npy":
            res[name] = np.load(filename)
        elif ext == "csv":
            res[name] = pd.read_csv(filename)
        elif ext == "yaml":
            with open(filename) as f:
                res[name] = yaml.safe_load(f)
        else:
            warnings.warn("File '%s' with unsupported extension will be ignored." % filename)

    return res


class Experiment(ABC):

    scenarios: Dict[str, Type["Experiment"]] = {}
    scenario_domains: Dict[str, Dict[str, Set[Any]]] = {}
    domains: Dict[str, Set[Any]] = {}
    _scenario: Optional[str] = None

    def __init__(self, id: Optional[str] = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._id = id if id is not None else "".join(random.choices(string.ascii_lowercase + string.digits, k=10))

    def __init_subclass__(cls: Type["Experiment"], scenario: str) -> None:
        # Register scenario under the given name.
        cls._scenario = scenario
        Experiment.scenarios[scenario] = cls

        # Extract domain of scenario.
        props = attribute.get_properties(cls)
        domain = dict((name, set(get_property_domain(cls, name))) for name in props.keys())
        Experiment.scenario_domains[scenario] = domain

        # Include the new domain into the total domain.
        for name, values in domain.items():
            Experiment.domains.setdefault(name, set()).update(values)

    @abstractmethod
    def run(self, **kwargs: Any) -> None:
        raise NotImplementedError()

    @abstractproperty
    def completed(self) -> bool:
        raise NotImplementedError()

    @abstractproperty
    def dataframe(self) -> DataFrame:
        raise NotImplementedError()

    @attribute
    def scenario(self) -> str:
        assert self.__class__._scenario is not None
        return self.__class__._scenario

    @attribute
    def id(self) -> str:
        return self._id

    @classmethod
    def get_instances(cls, **kwargs: Any) -> Iterable["Experiment"]:
        if cls == Experiment:
            for scenario in Experiment.scenarios.values():
                for experiment in scenario.get_instances(**kwargs):
                    yield experiment
        else:
            domains = []
            names = Experiment.domains.keys()
            for name in names:
                if name in kwargs:
                    domains.append([kwargs[name]])
                else:
                    domains.append(list(Experiment.domains[name]))
            for values in product(domains):
                attributes = dict((name, value) for (name, value) in zip(names, values) if value is not None)
                if cls.is_valid_config(**attributes):
                    yield cls(**attributes)

    @classmethod
    def is_valid_config(cls, **attributes: Any) -> bool:
        return True

    def save(self, path: str) -> None:
        if os.path.splitext(path)[1] != "":
            raise ValueError("The provided path '%s' is not a valid directory path." % path)
        os.makedirs(path, exist_ok=True)

        # Save attributes as a single yaml file.
        props = attribute.get_properties(self)
        attributes = dict((name, prop.fget(self) if prop.fget is not None else None) for (name, prop) in props.items())
        save_dict(attributes, path, "attributes")

        # Save results as separate files.
        props = result.get_properties(self)
        results = dict((name, prop.fget(self) if prop.fget is not None else None) for (name, prop) in props.items())
        save_dict(results, path, "results")

    @classmethod
    def from_dict(cls, source: Dict[str, Any]) -> "Experiment":
        return cls(**source)

    @classmethod
    def load(cls, path: str) -> "Experiment":
        if not os.path.isdir(path):
            raise ValueError("The provided path '%s' does not point to a directory." % path)

        attributes = load_dict(path, "attributes")
        results = load_dict(path, "results")
        return cls.from_dict({**attributes, **results})


class Study:
    def __init__(self, experiments: Sequence[Experiment], id: Optional[str] = None) -> None:
        self._experiments = experiments
        self._id = id if id is not None else datetime.now().strftime("Study-%Y-%m-%d-%H-%M-%S")

    def run(self, catch_exceptions: bool = True, **kwargs: Any) -> None:
        # TODO: Add progress bar.
        for experiment in self.experiments:
            try:
                experiment.run()
            except Exception as e:
                if catch_exceptions:
                    trace_output = traceback.format_exc()
                    print(trace_output)
                else:
                    raise e

    def save(self, path: str, experiment_path: Optional[str]) -> None:
        if experiment_path is None:
            experiment_path = "{id}"

        # Make directory that will contain the study.
        studypath = os.path.join(path, self.id)
        os.makedirs(studypath, exist_ok=True)

        # Iterate over all experiments and compute their target paths.
        attributes = re.findall(r"\{(.*?)\}", experiment_path)
        experiment_paths: Set[str] = set()
        for experiment in self.experiments:
            replacements = dict((name, get_property_value(experiment, name)) for name in attributes)
            exppath = experiment_path.format_map(replacements)
            if exppath in experiment_paths:
                raise ValueError(
                    "Provided experiment_path does not produce unique paths. The path '%s' caused a conflict." % exppath
                )
            experiment_paths.add(exppath)
            experiment.save(exppath)

    @classmethod
    def load(cls, path: str, id: Optional[str] = None) -> "Study":
        if id is None:
            id = os.path.basename(path)
        else:
            path = os.path.join(path, id)
        experiments: List[Experiment] = []
        for attpath in glob("**/attributes.yaml"):
            exppath = os.path.dirname(attpath)
            experiment = Experiment.load(exppath)
            experiments.append(experiment)
        return Study(experiments, id)

    @property
    def id(self) -> str:
        return self._id

    @property
    def completed(self) -> bool:
        return all(exp.completed for exp in self.experiments)

    @property
    def experiments(self) -> Sequence[Experiment]:
        return self._experiments

    def get_experiments(self, **attributes: Dict[str, Any]) -> Sequence[Experiment]:
        res: List[Experiment] = []
        for exp in self.experiments:
            if all(has_attribute_value(exp, name, value) for (name, value) in attributes.items()):
                res.append(exp)
        return res
