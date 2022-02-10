import datetime
import logging
import logging.handlers
import numpy as np
import os
import pandas as pd
import random
import re
import string
import sys
import threading
import time
import traceback
import warnings
import yaml

from abc import ABC, abstractmethod
from enum import Enum
from glob import glob
from inspect import signature
from io import TextIOBase, StringIO, SEEK_END
from itertools import product
from logging import Logger
from matplotlib.figure import Figure
from numpy import ndarray
from pandas import DataFrame
from ray.util.multiprocessing import Pool
from ray.util.queue import Queue
from shutil import copyfileobj
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    TypeVar,
    get_origin,
    get_args,
    overload,
    Union,
)


class PropertyTag(property):

    domain: Optional[Iterable] = None

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
    def __init__(
        self,
        fget: Optional[Callable[[Any], Any]] = None,
        fset: Optional[Callable[[Any, Any], None]] = None,
        fdel: Optional[Callable[[Any], None]] = None,
        doc: Optional[str] = None,
        domain: Optional[Iterable] = None,
    ) -> None:
        super().__init__(fget, fset, fdel, doc)
        self.domain = domain

    def __call__(
        self,
        fget: Optional[Callable[[Any], Any]] = None,
        fset: Optional[Callable[[Any, Any], None]] = None,
        fdel: Optional[Callable[[Any], None]] = None,
        doc: Optional[str] = None,
    ) -> "attribute":
        if fget is None:
            fget = self.fget
        if fset is None:
            fset = self.fset
        if fdel is None:
            fdel = self.fdel
        if doc is None:
            doc = self.__doc__
        return type(self)(fget, fset, fdel, doc, self.domain)

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


def get_property_and_getter(target: object, name: str) -> Tuple[property, Callable]:
    member = getattr(target, name)
    if not isinstance(member, property):
        raise ValueError("The specified member '%s' is not a property." % name)
    if member.fget is None:
        raise ValueError("The specified member '%s' does not have a getter." % name)
    return member, member.fget


def get_property_type(target: object, name: str) -> Optional[Type]:
    _, getter = get_property_and_getter(target, name)
    sign = signature(getter)
    return sign.return_annotation if isinstance(sign.return_annotation, type) else None


def get_property_domain(target: object, name: str) -> List[Any]:
    prop, getter = get_property_and_getter(target, name)
    sign = signature(getter)
    enum = extract_enumtype(sign.return_annotation)
    if isinstance(prop, PropertyTag) and prop.domain is not None:
        return list(prop.domain)
    elif enum is None:
        return [None]
    else:
        return list(enum.__members__.values())


def get_property_helpstring(target: object, name: str) -> Optional[str]:
    prop, getter = get_property_and_getter(target, name)
    return getter.__doc__


def get_property_default(target: object, name: str) -> Optional[Any]:
    member = getattr(target, "__init__")
    sign = signature(member)
    param = sign.parameters.get(name, None)
    if param is not None and param.default is not param.empty:
        return param.default
    else:
        return None


def get_property_value(target: object, name: str) -> Any:
    member = getattr(target, name)
    if isinstance(target, type):
        if isinstance(member, property) and member.fget is not None:
            return member.fget(target)
        else:
            return None
    else:
        return member


def has_attribute_value(target: object, name: str, value: Any, ignore_none: bool = True) -> bool:
    target_value = get_property_value(target, name)
    if not isinstance(value, Iterable):
        value = [value]
    if ignore_none:
        return target_value is None or target_value in value
    else:
        return target_value in value


def save_dict(source: Dict[str, Any], dirpath: str, basename: str) -> None:
    basedict: Dict[str, Any] = dict((k, v) for (k, v) in source.items() if type(v) in [int, float, bool, str])
    basedict.update(dict((k, v.value) for (k, v) in source.items() if isinstance(v, Enum)))
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
            elif isinstance(data, Figure):
                filename = os.path.join(dirpath, ".".join([basename, name, "pdf"]))
                data.savefig(fname=filename)
                filename = os.path.join(dirpath, ".".join([basename, name, "png"]))
                data.savefig(fname=filename)
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

        if name == "":
            continue

        if ext == ".npy":
            res[name] = np.load(path)
        elif ext == ".csv":
            res[name] = pd.read_csv(path)
        elif ext == ".yaml":
            with open(path) as f:
                res[name] = yaml.safe_load(f)
        else:
            warnings.warn("File '%s' with unsupported extension '%s' will be ignored." % (path, ext))

    return res


class ProgressEventType(Enum):
    START = "start"
    UPDATE = "update"
    CLOSE = "close"


class ProgressEvent:
    def __init__(self, type: ProgressEventType, id: str, **kwargs: Any) -> None:
        self.type = type
        self.id = id
        self.kwargs = kwargs


class Progress:

    pbars: Dict[str, tqdm] = {}

    def __init__(self, queue: Optional[Queue] = None, id: Optional[str] = None) -> None:
        self._queue = queue
        self._id = id if id is not None else "".join(random.choices(string.ascii_lowercase + string.digits, k=10))

    def start(self, total: Optional[int] = None, desc: Optional[str] = None) -> None:
        self._submit(ProgressEvent(ProgressEventType.START, self._id, total=total, desc=desc))

    def update(self, n: int = 1) -> None:
        self._submit(ProgressEvent(ProgressEventType.UPDATE, self._id, n=n))

    def close(self) -> None:
        self._submit(ProgressEvent(ProgressEventType.CLOSE, self._id))

    def _submit(self, event: ProgressEvent) -> None:
        if self._queue is None:
            self.handle(event)
        else:
            self._queue.put(event)

    def new(self, id: Optional[str] = None) -> "Progress":
        return Progress(self._queue, id)

    @property
    def queue(self) -> Optional[Queue]:
        return self._queue

    @queue.setter
    def queue(self, value: Optional[Queue]) -> None:
        self._queue = value

    @classmethod
    def refresh(cls: Type["Progress"]) -> None:
        for pbar in cls.pbars.values():
            pbar.refresh()

    @classmethod
    def handle(cls: Type["Progress"], event: ProgressEvent) -> None:
        if event.type == ProgressEventType.START:
            cls.pbars[event.id] = tqdm(desc=event.kwargs["desc"], total=event.kwargs["total"])
        elif event.type == ProgressEventType.UPDATE:
            cls.pbars[event.id].update(event.kwargs["n"])
        elif event.type == ProgressEventType.CLOSE:
            cls.pbars[event.id].close()
            del cls.pbars[event.id]
            # Ensure that bars get redrawn properly after they reshuffle due to closure of one of them.
            cls.refresh()


class Scenario(ABC):

    scenarios: Dict[str, Type["Scenario"]] = {}
    scenario_domains: Dict[str, Dict[str, Set[Any]]] = {}
    attribute_domains: Dict[str, Set[Any]] = {}
    attribute_helpstrings: Dict[str, Optional[str]] = {}
    attribute_types: Dict[str, Optional[type]] = {}
    attribute_defaults: Dict[str, Optional[Any]] = {}
    _scenario: Optional[str] = None

    def __init__(self, id: Optional[str] = None, logstream: Optional[TextIOBase] = None, **kwargs: Any) -> None:
        super().__init__()
        self._id = id if id is not None else "".join(random.choices(string.ascii_lowercase + string.digits, k=10))
        self._logstream = logstream if logstream is not None else StringIO()
        self._progress = Progress(id=self._id)
        self._attributes: Optional[Dict[str, Any]] = None

    def __init_subclass__(cls: Type["Scenario"], id: str) -> None:
        # Register scenario under the given name.
        cls._scenario = id
        Scenario.scenarios[id] = cls
        assert isinstance(Scenario.scenario, PropertyTag)
        assert isinstance(Scenario.scenario.domain, set)
        Scenario.scenario.domain.add(id)

        # Extract domain of scenario.
        props = attribute.get_properties(cls)
        domain = dict((name, set(get_property_domain(cls, name))) for name in props.keys())
        Scenario.scenario_domains[id] = domain

        # Include the new domain into the total domain.
        for name, values in domain.items():
            Scenario.attribute_domains.setdefault(name, set()).update(values)

        # Extract types of the scenario attributes.
        types = dict((name, get_property_type(cls, name)) for name in props.keys())
        Scenario.attribute_types.update(types)

        # Extract helpstrings of the scenario attributes.
        helpstrings = dict((name, get_property_helpstring(cls, name)) for name in props.keys())
        Scenario.attribute_helpstrings.update(helpstrings)

        # Extract types of the scenario attributes.
        defaults = dict((name, get_property_default(cls, name)) for name in props.keys())
        Scenario.attribute_defaults.update(defaults)

    def run(self, progress_bar: bool = True, console_log: bool = True) -> None:
        # Set up logging.
        formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
        fh = logging.StreamHandler(self._logstream)
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        ch: Optional[logging.Handler] = None
        if console_log:
            ch = logging.StreamHandler(sys.stdout)
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)

        self.logger.info("Run started for scenario: %s" % str(self))
        timestart = time.time()

        self._run(progress_bar=progress_bar)

        duration = datetime.timedelta(seconds=int(time.time() - timestart))
        # duration = datetime.time(0, 0, int(time.time() - timestart)).strftime("%H:%M:%S")
        self.logger.info("Run completed. Duration: %s" % str(duration))

        # Clear logging handlers.
        self.logger.removeHandler(fh)
        if ch is not None:
            self.logger.removeHandler(ch)

    @abstractmethod
    def _run(self, progress_bar: bool = True, **kwargs: Any) -> None:
        raise NotImplementedError()

    @property
    @abstractmethod
    def completed(self) -> bool:
        raise NotImplementedError()

    @property
    @abstractmethod
    def dataframe(self) -> DataFrame:
        raise NotImplementedError()

    @attribute(domain=set())
    def scenario(self) -> str:
        """Type of scenario."""
        if self._scenario is None:
            raise ValueError("Cannot call this on an abstract class instance.")
        return self._scenario

    @attribute
    def id(self) -> str:
        """A unique identifier of the scenario."""
        return self._id

    @property
    def progress(self) -> Progress:
        return self._progress

    @property
    def logger(self) -> Logger:
        return logging.getLogger(self._id)

    @property
    def logstream(self) -> Optional[TextIOBase]:
        return self._logstream

    @property
    def log(self) -> str:
        result = ""
        self._logstream.seek(0)
        result = "\n".join(self._logstream.readlines())
        self._logstream.seek(0, SEEK_END)
        return result

    @property
    def attributes(self) -> Dict[str, Any]:
        if self._attributes is None:
            props = attribute.get_properties(self.__class__)
            self._attributes = dict((name, set(get_property_value(self, name))) for name in props.keys())
        return self._attributes

    @classmethod
    def get_instances(cls, **kwargs: Any) -> Iterable["Scenario"]:
        if cls == Scenario:
            for scenario in Scenario.scenarios.values():
                for instance in scenario.get_instances(**kwargs):
                    yield instance
        else:
            domains = []
            names = Scenario.attribute_domains.keys()
            for name in names:
                if name in kwargs and kwargs[name] is not None:
                    domain = kwargs[name]
                    if not isinstance(domain, Iterable):
                        domain = [domain]
                    domains.append(list(domain))
                else:
                    domains.append(list(Scenario.attribute_domains[name]))
            for values in product(*domains):
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

        # Write a log file.
        logpath = os.path.join(path, "scenario.log")
        with open(logpath, "w") as f:
            self._logstream.seek(0)
            copyfileobj(self._logstream, f)
            self._logstream.seek(0, SEEK_END)

        # Save attributes as a single yaml file.
        props = attribute.get_properties(type(self))
        attributes = dict((name, prop.fget(self) if prop.fget is not None else None) for (name, prop) in props.items())
        save_dict(attributes, path, "attributes")

        # Save results as separate files.
        props = result.get_properties(type(self))
        results = dict((name, prop.fget(self) if prop.fget is not None else None) for (name, prop) in props.items())
        save_dict(results, path, "results")

    def __str__(self) -> str:
        props = attribute.get_properties(type(self))
        attributes = dict((name, prop.fget(self) if prop.fget is not None else None) for (name, prop) in props.items())
        for k, v in attributes.items():
            if isinstance(v, Enum):
                attributes[k] = v.value
        return "(%s)" % ", ".join(["%s=%s" % (str(k), str(v)) for (k, v) in attributes.items()])

    @classmethod
    def from_dict(cls, source: Dict[str, Any]) -> "Scenario":
        scenario_id = source["scenario"]
        scenario_cls = cls.scenarios[scenario_id]
        return scenario_cls(**source)

    @classmethod
    def load(cls, path: str) -> "Scenario":
        if not os.path.isdir(path):
            raise ValueError("The provided path '%s' does not point to a directory." % path)

        # Load the log file.
        logpath = os.path.join(path, "study.log")
        logstream: Optional[TextIOBase] = None
        if os.path.isfile(logpath):
            with open(logpath, "r") as f:
                logstream = StringIO(f.read())

        attributes = load_dict(path, "attributes")
        results = load_dict(path, "results")
        kwargs = {"logstream": logstream}
        return cls.from_dict({**attributes, **results, **kwargs})


V = TypeVar("V")


def get_value(obj: Any, key: str) -> Any:
    if hasattr(obj, key):
        return getattr(obj, key)
    else:
        return None


class Table(Sequence[V]):
    def __init__(self, data: Sequence[V], attributes: List[str] = [], key: Optional[str] = None):
        self._data = data
        self._attributes = attributes
        self._key = key

    @overload
    def __getitem__(self, index: int) -> V:
        return self._data.__getitem__(index)

    @overload
    def __getitem__(self, index: slice) -> Sequence[V]:
        return self._data.__getitem__(index)

    def __getitem__(self, index: Union[int, slice]) -> Union[V, Sequence[V]]:
        return self._data.__getitem__(index)

    def __len__(self) -> int:
        return self._data.__len__()

    @property
    def df(self):
        df = pd.DataFrame.from_dict({a: [get_value(x, a) for x in self._data] for a in self._attributes})
        if self._key is not None:
            df.set_index(self._key, inplace=True)
        return df

    def __repr__(self) -> str:
        return self.df.__repr__()

    def _repr_html_(self) -> Optional[str]:
        return self.df._repr_html_()


DEFAULT_OUTPUT_PATH = os.path.join("var", "results")
DEFAULT_SCENARIO_PATH_FORMAT = "{id}"


class Study:
    def __init__(
        self,
        scenarios: Sequence[Scenario],
        id: Optional[str] = None,
        outpath: str = DEFAULT_OUTPUT_PATH,
        scenario_path_format: str = DEFAULT_SCENARIO_PATH_FORMAT,
        logstream: Optional[TextIOBase] = None,
    ) -> None:
        self._scenarios = scenarios
        self._id = id if id is not None else datetime.datetime.now().strftime("Study-%Y-%m-%d-%H-%M-%S")
        self._outpath = outpath
        self._scenario_path_format = scenario_path_format
        self._logstream = logstream if logstream is not None else StringIO()
        self._logger = logging.getLogger(self._id)
        self._verify_scenario_path(scenario_path_format, scenarios)

    @staticmethod
    def _get_scenario_runner(
        queue: Optional[Queue] = None,
        catch_exceptions: bool = True,
        progress_bar: bool = True,
        console_log: bool = True,
    ) -> Callable[[Scenario], Scenario]:
        def _scenario_runner(scenario: Scenario) -> Scenario:
            try:
                scenario.progress.queue = queue
                scenario.logger.setLevel(logging.DEBUG)
                qh: Optional[logging.Handler] = None
                ch: Optional[logging.Handler] = None
                if queue is not None:
                    qh = logging.handlers.QueueHandler(queue)  # type: ignore
                    scenario.logger.addHandler(qh)
                elif console_log:
                    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s")
                    ch = logging.StreamHandler(sys.stdout)
                    ch.setFormatter(formatter)
                    scenario.logger.addHandler(ch)

                if queue is not None:
                    scenario.run(progress_bar=progress_bar, console_log=False)
                else:
                    with logging_redirect_tqdm(loggers=[scenario.logger]):
                        scenario.run(progress_bar=progress_bar, console_log=False)
            except Exception as e:
                if catch_exceptions:
                    trace_output = traceback.format_exc()
                    scenario.logger.error(trace_output)
                else:
                    raise e
            finally:
                scenario.progress.queue = None
                if qh is not None:
                    scenario.logger.removeHandler(qh)
                if ch is not None:
                    scenario.logger.removeHandler(ch)
            return scenario

        return _scenario_runner

    @staticmethod
    def _status_monitor(queue: Queue, logger: Logger) -> None:
        while True:
            record: Optional[Union[logging.LogRecord, ProgressEvent]] = queue.get()
            if record is None:
                break
            if isinstance(record, ProgressEvent):
                Progress.handle(record)
            else:
                # logger = logging.getLogger(record.name)
                logger.handle(record)
            Progress.refresh()

    def run(
        self,
        catch_exceptions: bool = True,
        progress_bar: bool = True,
        console_log: bool = True,
        parallel: bool = True,
        ray_address: Optional[str] = None,
        eagersave: bool = True,
        **kwargs: Any
    ) -> None:

        # Set up logging.
        formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s")
        fh = logging.StreamHandler(self._logstream)
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        ch: Optional[logging.Handler] = None
        if console_log:
            ch = logging.StreamHandler(sys.stdout)
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)

        # Set up progress bar.
        # pbar = None if not progress_bar else tqdm(total=len(self.scenarios), desc="Scenarios", position=0)
        queue = Queue() if parallel else None
        pbar = None if not progress_bar else Progress(queue, id=self.id)
        if pbar is not None:
            pbar.start(total=len(self.scenarios), desc="Scenarios")

        with logging_redirect_tqdm(loggers=[self.logger]):
            # for scenario in self.scenarios:
            #     try:
            #         scenario.run(logger=self._logger, progress_bar=progress_bar, **kwargs)
            #     except Exception as e:
            #         if catch_exceptions:
            #             trace_output = traceback.format_exc()
            #             print(trace_output)
            #         else:
            #             raise e
            #     finally:
            #         if pbar is not None:
            #             pbar.update(1)

            scenarios = []
            runner = Study._get_scenario_runner(queue, catch_exceptions, progress_bar, console_log)
            if parallel:
                monitor = threading.Thread(target=Study._status_monitor, args=(queue, self.logger))
                monitor.start()
                pool = Pool(ray_address=ray_address)
                for scenario in pool.imap_unordered(runner, self.scenarios):
                    scenarios.append(scenario)
                    if pbar is not None:
                        pbar.update(1)
                    if eagersave:
                        self.save_scenario(scenario)

            else:
                for scenario in map(runner, self.scenarios):
                    scenarios.append(scenario)
                    if pbar is not None:
                        pbar.update(1)
                    if eagersave:
                        self.save_scenario(scenario)

            self._scenarios = scenarios

            if pbar is not None:
                pbar.close()

            if parallel:
                assert queue is not None and monitor is not None
                queue.put(None)
                monitor.join()

        # Clear logging handlers.
        self.logger.removeHandler(fh)
        if ch is not None:
            self.logger.removeHandler(ch)

    @staticmethod
    def _verify_scenario_path(scenario_path_format: str, scenarios: Iterable[Scenario]) -> None:
        attributes = re.findall(r"\{(.*?)\}", scenario_path_format)
        scenario_paths: Set[str] = set()
        for scenario in scenarios:
            replacements = dict((name, get_property_value(scenario, name)) for name in attributes)
            exppath = scenario_path_format.format_map(replacements)
            if exppath in scenario_paths:
                raise ValueError(
                    "Provided scenario_path does not produce unique paths. The path '%s' caused a conflict." % exppath
                )
            scenario_paths.add(exppath)

    def save_scenario(self, scenario: Scenario) -> str:
        if self.path is None:
            raise ValueError("This scenario has no output path.")
        attributes = re.findall(r"\{(.*?)\}", self._scenario_path_format)
        replacements = dict((name, get_property_value(scenario, name)) for name in attributes)
        exppath = self._scenario_path_format.format_map(replacements)
        full_exppath = os.path.join(self.path, "scenarios", exppath)
        scenario.save(full_exppath)
        return full_exppath

    def save(self, save_scenarios: bool = True) -> None:

        # Make directory that will contain the study.
        os.makedirs(self.path, exist_ok=True)

        # Write a marker file.
        markerpath = os.path.join(self.path, "study.yml")
        with open(markerpath, "w") as f:
            yaml.safe_dump({"id": self.id, "scenario_path_format": self.scenario_path_format}, f)

        # Write a log file.
        logpath = os.path.join(self.path, "study.log")
        with open(logpath, "w") as f:
            self._logstream.seek(0)
            copyfileobj(self._logstream, f)
            self._logstream.seek(0, SEEK_END)

        # Verify that the provided scenario path is unique enough.
        # self._verify_scenario_path(self._scenario_path_format, self.scenarios)

        # Iterate over all scenarios and compute their target paths.
        # attributes = re.findall(r"\{(.*?)\}", scenario_path)
        # scenario_paths: Set[str] = set()
        if save_scenarios:
            for scenario in self.scenarios:
                self.save_scenario(scenario)
                # replacements = dict((name, get_property_value(scenario, name)) for name in attributes)
                # exppath = scenario_path.format_map(replacements)
                # if exppath in scenario_paths:
                #     raise ValueError(
                #         "Provided scenario_path does not produce unique paths. The path '%s' caused a conflict."
                #         % exppath
                #     )
                # scenario_paths.add(exppath)
                # full_exppath = os.path.join(self.path, "scenarios", exppath)
                # scenario.save(full_exppath)

    @classmethod
    def load(cls, path: str, id: Optional[str] = None) -> "Study":
        outpath = path
        if id is None:
            id = os.path.basename(path)
            outpath = os.path.dirname(path)
        else:
            path = os.path.join(path, id)

        # Load the marker file.
        markerpath = os.path.join(path, "study.yml")
        with open(markerpath) as f:
            metadata = yaml.safe_load(f)
            if metadata["id"] != id:
                raise ValueError(
                    "ID mismatch between the provided '%s' and the encountered '%s'." % (id, metadata["id"])
                )

        # Load the log file.
        logpath = os.path.join(path, "study.log")
        logstream: Optional[TextIOBase] = None
        if os.path.isfile(logpath):
            with open(logpath, "r") as f:
                logstream = StringIO(f.read())

        # Load all scenarios.
        scenarios: List[Scenario] = []
        for attpath in glob(os.path.join(path, "**/attributes.yaml"), recursive=True):
            exppath = os.path.dirname(attpath)
            scenario = Scenario.load(exppath)
            scenarios.append(scenario)

        # Reconstruct study and return it.
        result = Study(
            scenarios=scenarios,
            id=id,
            outpath=outpath,
            scenario_path_format=metadata["scenario_path_format"],
            logstream=logstream,
        )
        return result

    @classmethod
    def isstudy(cls, path: str) -> bool:
        return os.path.isfile(os.path.join(path, "study.yml"))

    @property
    def id(self) -> str:
        return self._id

    @property
    def path(self) -> str:
        return os.path.join(self._outpath, self._id)

    @property
    def scenario_path_format(self) -> str:
        return self._scenario_path_format

    @property
    def logger(self) -> Logger:
        return self._logger

    @property
    def logstream(self) -> Optional[TextIOBase]:
        return self._logstream

    @property
    def log(self) -> str:
        result = ""
        self._logstream.seek(0)
        result = "\n".join(self._logstream.readlines())
        self._logstream.seek(0, SEEK_END)
        return result

    @property
    def completed(self) -> bool:
        return all(exp.completed for exp in self.scenarios if isinstance(exp, Scenario))

    @property
    def scenarios(self) -> Table[Scenario]:
        attributes = list(Scenario.attribute_domains.keys())
        return Table[Scenario](self._scenarios, attributes, "id")

    @property
    def dataframe(self) -> DataFrame:
        return pd.concat([scenario.dataframe for scenario in self.scenarios], ignore_index=True)

    def get_scenarios(self, **attributes: Dict[str, Any]) -> Sequence[Scenario]:
        res: List[Scenario] = []
        for exp in self.scenarios:
            if all(has_attribute_value(exp, name, value) for (name, value) in attributes.items()):
                res.append(exp)
        return res


class Report(ABC):

    reports: Dict[str, Type["Report"]] = {}
    _report: Optional[str] = None

    def __init__(
        self, study: Study, id: Optional[str] = None, attributes: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> None:
        super().__init__()
        self._study = study
        self._id = id if id is not None else "".join(random.choices(string.ascii_lowercase + string.digits, k=10))
        self._attributes: Dict[str, Any] = {} if attributes is None else attributes

    def __init_subclass__(cls: Type["Report"], id: str) -> None:
        cls._report = id
        cls.reports[id] = cls

    @property
    def report(self) -> str:
        if self._report is None:
            raise ValueError("Cannot call this on an abstract class instance.")
        return self._report

    @property
    def id(self) -> str:
        """A unique identifier of the report."""
        return self._id

    @property
    def attributes(self) -> Dict[str, Any]:
        return self._attributes

    @property
    def study(self) -> Study:
        return self._study

    @abstractmethod
    def generate(self) -> None:
        raise NotImplementedError()

    def save(self, path: Optional[str] = None, use_attributes: bool = True, use_id: bool = False) -> None:
        if path is None:
            path = os.path.join(self._study.path, "reports")
        if use_attributes:
            attributes = [self._attributes[key] for key in sorted(self.attributes.keys())]
            path = os.path.join(path, *attributes)
        if use_id:
            path = os.path.join(path, self._id)
        if os.path.splitext(path)[1] != "":
            raise ValueError("The provided path '%s' is not a valid directory path." % path)
        os.makedirs(path, exist_ok=True)

        # Save results as separate files.
        props = result.get_properties(type(self))
        results = dict((name, prop.fget(self) if prop.fget is not None else None) for (name, prop) in props.items())
        save_dict(results, path, "report")