import argparse
import collections.abc
import datetime
import gevent
import gevent.signal
import logging
import logging.handlers
import numpy as np
import os
import pandas as pd
import pickle
import random
import re
import signal
import socket
import string
import subprocess
import sys
import threading
import time
import traceback
import warnings
import yaml
import zerorpc

from abc import ABC, abstractmethod
from enum import Enum
from glob import glob
from inspect import signature
from io import TextIOBase, StringIO, SEEK_END
from itertools import product
from logging import Logger
from matplotlib.figure import Figure
from multiprocessing import Process, Queue as MultiprocessingQueue
from numpy import ndarray
from pandas import DataFrame
from ray.util.multiprocessing import Pool
from ray.util.queue import Queue as RayQueue
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
    Protocol,
)


class Queue(Protocol):
    def get(self, block: bool = True, timeout: Optional[float] = None) -> Any:
        ...

    def put(self, item: Any, block: bool = True, timeout: Optional[float] = None) -> None:
        ...


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
        domain: Optional[Union[Iterable, Dict]] = None,
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
    a = sign.return_annotation
    if get_origin(a) in [collections.abc.Sequence, collections.abc.Iterable, list, set] and len(get_args(a)) > 0:
        a = get_args(a)[0]
    if get_origin(a) == Union:
        a = next(x for x in get_args(a) if x is not None)
    return a if isinstance(a, type) else None


def get_property_domain(target: object, name: str) -> List[Any]:
    prop, getter = get_property_and_getter(target, name)
    sign = signature(getter)
    enum = extract_enumtype(sign.return_annotation)
    if isinstance(prop, PropertyTag) and prop.domain is not None:
        return list(prop.domain.keys()) if isinstance(prop.domain, dict) else list(prop.domain)
    elif sign.return_annotation is bool:
        return [False, True]
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


def get_property_isiterable(target: object, name: str) -> bool:
    _, getter = get_property_and_getter(target, name)
    sign = signature(getter)
    a = sign.return_annotation
    return get_origin(a) in [collections.abc.Sequence, collections.abc.Iterable, list, set]


def has_attribute_value(target: object, name: str, value: Any, ignore_none: bool = True) -> bool:
    target_value = get_property_value(target, name)
    if not isinstance(value, Iterable):
        value = [value]
    if ignore_none:
        return target_value is None or value == [None] or target_value in value
    else:
        return target_value in value


def make_type_parser(target: Optional[type]) -> Callable[[str], Any]:
    def parser(source: str) -> Any:
        if target is None:
            return source
        result: Any = source
        if issubclass(target, bool):
            result = result in ["True", "true", "T", "t", "Yes", "yes", "y"]
        elif issubclass(target, int):
            result = int(result)
        elif issubclass(target, float):
            result = float(result)
        elif issubclass(target, Enum):
            result = target(result)
        return result

    return parser


def add_dynamic_arguments(
    parser: argparse.ArgumentParser,
    targets: Iterable[type],
    all_iterable: bool = False,
    single_instance: bool = False,
) -> None:

    attribute_domains: Dict[str, Set[Any]] = {}
    attribute_helpstrings: Dict[str, Optional[str]] = {}
    attribute_types: Dict[str, Optional[type]] = {}
    attribute_defaults: Dict[str, Optional[Any]] = {}
    attribute_isiterable: Dict[str, bool] = {}

    for cls in targets:
        # Extract domain of scenario.
        props = attribute.get_properties(cls)
        domain = dict((name, set(get_property_domain(cls, name))) for name in props.keys())

        # Include the new domain into the total domain.
        for name, values in domain.items():
            attribute_domains.setdefault(name, set()).update(values)

        # Extract types of the scenario attributes.
        types = dict((name, get_property_type(cls, name)) for name in props.keys())
        attribute_types.update(types)

        # Extract helpstrings of the scenario attributes.
        helpstrings = dict((name, get_property_helpstring(cls, name)) for name in props.keys())
        attribute_helpstrings.update(helpstrings)

        # Extract types of the scenario attributes.
        defaults = dict((name, get_property_default(cls, name)) for name in props.keys())
        attribute_defaults.update(defaults)

        # Set all attributes to be iterable when passed to get_instances.
        isiterable = dict((name, True if all_iterable else get_property_isiterable(cls, name)) for name in props.keys())
        attribute_isiterable.update(isiterable)

    for name in attribute_domains:
        default = attribute_defaults[name]
        attribute_domain: Optional[List] = [x.value if isinstance(x, Enum) else x for x in attribute_domains[name]]
        if attribute_domain == [None]:
            attribute_domain = None
        helpstring = attribute_helpstrings[name] or ("Scenario " + name + ".")
        if default is None:
            helpstring += " Default: [all]" if not single_instance else ""
        else:
            helpstring += " Default: %s" % str(default)
        parser.add_argument(
            "--%s" % name.replace("_", "-"),
            help=helpstring,
            type=make_type_parser(attribute_types[name]),
            choices=attribute_domain,
            nargs=(1 if single_instance else "+") if attribute_isiterable[name] else None,  # type: ignore
        )


def save_dict(source: Dict[str, Any], dirpath: str, basename: str, saveonly: Optional[Sequence[str]] = None) -> None:
    basedict: Dict[str, Any] = dict((k, v) for (k, v) in source.items() if type(v) in [int, float, bool, str])
    basedict.update(dict((k, v.value) for (k, v) in source.items() if isinstance(v, Enum)))
    if len(basedict) > 0:
        with open(os.path.join(dirpath, ".".join([basename, "yaml"])), "w") as f:
            yaml.safe_dump(basedict, f)

    for name, data in source.items():
        if data is None:
            continue
        if name not in basedict:
            if saveonly is not None and len(saveonly) > 0 and name not in saveonly:
                continue

            if isinstance(data, ndarray):
                filename = os.path.join(dirpath, ".".join([basename, name, "npy"]))
                np.save(filename, data)
            elif isinstance(data, DataFrame):
                filename = os.path.join(dirpath, ".".join([basename, name, "csv"]))
                data.to_csv(filename)
            elif isinstance(data, dict):
                filename = os.path.join(dirpath, ".".join([basename, name, "yaml"]))
                with open(filename, "w") as f:
                    yaml.safe_dump(data, f)
            elif isinstance(data, Figure):
                extra_artists = tuple(data.legends) + tuple(data.texts)
                filename = os.path.join(dirpath, ".".join([basename, name, "pdf"]))
                data.savefig(fname=filename, bbox_extra_artists=extra_artists, bbox_inches="tight")
                filename = os.path.join(dirpath, ".".join([basename, name, "png"]))
                data.savefig(fname=filename, bbox_extra_artists=extra_artists, bbox_inches="tight")
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


class Backend(str, Enum):
    LOCAL = "local"
    RAY = "ray"
    SLURM = "slurm"


class Progress:
    class Event:
        class Type(Enum):
            START = "start"
            UPDATE = "update"
            CLOSE = "close"

        def __init__(self, type: "Progress.Event.Type", id: str, **kwargs: Any) -> None:
            self.type = type
            self.id = id
            self.kwargs = kwargs

    pbars: Dict[str, tqdm] = {}

    def __init__(self, queue: Optional[Queue] = None, id: Optional[str] = None, pickled: bool = False) -> None:
        self._queue = queue
        self._pickled = pickled
        self._id = id if id is not None else "".join(random.choices(string.ascii_lowercase + string.digits, k=10))

    def start(self, total: Optional[int] = None, desc: Optional[str] = None) -> None:
        self._submit(Progress.Event(Progress.Event.Type.START, self._id, total=total, desc=desc))

    def update(self, n: int = 1) -> None:
        self._submit(Progress.Event(Progress.Event.Type.UPDATE, self._id, n=n))

    def close(self) -> None:
        self._submit(Progress.Event(Progress.Event.Type.CLOSE, self._id))

    def _submit(self, event: "Progress.Event") -> None:
        if self._queue is None:
            self.handle(event)
        else:
            payload: Union["Progress.Event", bytes] = event
            if self._pickled:
                payload = pickle.dumps(payload)
            self._queue.put(payload)

    def new(self, id: Optional[str] = None) -> "Progress":
        return Progress(self._queue, id)

    @property
    def queue(self) -> Optional[Queue]:
        return self._queue

    @queue.setter
    def queue(self, value: Optional[Queue]) -> None:
        self._queue = value

    @property
    def pickled(self) -> bool:
        return self._pickled

    @pickled.setter
    def pickled(self, value: bool) -> None:
        self._pickled = value

    @classmethod
    def refresh(cls: Type["Progress"]) -> None:
        for pbar in cls.pbars.values():
            pbar.refresh()

    @classmethod
    def handle(cls: Type["Progress"], event: "Progress.Event") -> None:
        if event.type == Progress.Event.Type.START:
            cls.pbars[event.id] = tqdm(desc=event.kwargs["desc"], total=event.kwargs["total"])
        elif event.type == Progress.Event.Type.UPDATE:
            cls.pbars[event.id].update(event.kwargs["n"])
        elif event.type == Progress.Event.Type.CLOSE:
            cls.pbars[event.id].close()
            del cls.pbars[event.id]
            # Ensure that bars get redrawn properly after they reshuffle due to closure of one of them.
            cls.refresh()


class Scenario(ABC):

    scenarios: Dict[str, Type["Scenario"]] = {}
    attribute_names: Set[str] = set()
    _scenario: Optional[str] = None

    def __init__(self, id: Optional[str] = None, logstream: Optional[TextIOBase] = None, **kwargs: Any) -> None:
        super().__init__()
        self._id = id if id is not None else "".join(random.choices(string.ascii_lowercase + string.digits, k=10))
        self._logstream = logstream if logstream is not None else StringIO()
        self._progress = Progress(id=self._id)
        self._attributes: Optional[Dict[str, Any]] = None

    def __init_subclass__(cls: Type["Scenario"], id: Optional[str] = None, abstract: bool = False) -> None:
        if abstract or id is None:
            return

        # Register scenario under the given name.
        cls._scenario = id
        Scenario.scenarios[id] = cls
        assert isinstance(Scenario.scenario, PropertyTag)
        assert isinstance(Scenario.scenario.domain, set)
        Scenario.scenario.domain.add(id)

        # Extract scenario attribute names.
        props = attribute.get_properties(cls)
        cls.attribute_names = set(props.keys())
        Scenario.attribute_names.update(cls.attribute_names)

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
            self._attributes = dict((name, get_property_value(self, name)) for name in props.keys())
        return self._attributes

    @property
    def keyword_replacements(self) -> Dict[str, str]:
        return {}

    @classmethod
    def get_instances(cls, **kwargs: Any) -> Iterable["Scenario"]:
        if cls == Scenario:
            for id, scenario in Scenario.scenarios.items():
                if kwargs.get("scenario", None) is None or id in kwargs["scenario"]:
                    for instance in scenario.get_instances(**kwargs):
                        yield instance
        else:
            domains = []
            names = cls.attribute_names
            for name in names:
                if name in kwargs and kwargs[name] is not None:
                    domain = kwargs[name]
                    if not isinstance(domain, Iterable):
                        domain = [domain]
                    domains.append(list(domain))
                else:
                    domains.append(get_property_domain(cls, name))
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
        logpath = os.path.join(path, "scenario.log")
        logstream: Optional[TextIOBase] = None
        if os.path.isfile(logpath):
            with open(logpath, "r") as f:
                logstream = StringIO(f.read())

        attributes = load_dict(path, "attributes")
        results = load_dict(path, "results")
        kwargs = {"logstream": logstream}
        return cls.from_dict({**attributes, **results, **kwargs})

    @classmethod
    def isscenario(cls, path: str) -> bool:
        attributes_filename = os.path.join(path, ".".join(["attributes", "yaml"]))
        return os.path.isdir(path) and os.path.isfile(attributes_filename)  # TODO: Refine this check.

    def is_match(self, other: "Scenario") -> bool:
        other_attributes = other.attributes
        return all(other_attributes.get(k, None) == v for (k, v) in self.attributes.items() if k != "id")


class ScenarioEvent:
    class Type(Enum):
        STARTED = "started"
        COMPLETED = "completed"

    def __init__(self, type: "ScenarioEvent.Type", id: str, **kwargs: Any) -> None:
        self.type = type
        self.id = id
        self.kwargs = kwargs


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


class QueueHandler(logging.handlers.QueueHandler):
    def __init__(self, queue, pickled: bool = False):
        logging.handlers.QueueHandler.__init__(self, queue)
        self.pickled = pickled

    def enqueue(self, record):
        if self.pickled:
            record = pickle.dumps(record)
        self.queue.put_nowait(record)


def get_scenario_runner(
    queue: Optional[Queue] = None,
    catch_exceptions: bool = True,
    progress_bar: bool = True,
    console_log: bool = True,
    rerun: bool = False,
    pickled_queue: bool = False,
) -> Callable[[Scenario], Scenario]:
    def _scenario_runner(scenario: Scenario) -> Scenario:
        try:
            scenario.progress.queue = queue
            scenario.progress.pickled = pickled_queue
            scenario.logger.setLevel(logging.DEBUG)
            qh: Optional[logging.Handler] = None
            ch: Optional[logging.Handler] = None
            if queue is not None:
                qh = QueueHandler(queue, pickled=pickled_queue)  # type: ignore
                scenario.logger.addHandler(qh)
                payload: Union["ScenarioEvent", bytes] = ScenarioEvent(ScenarioEvent.Type.STARTED, id=scenario.id)
                if pickled_queue:
                    payload = pickle.dumps(payload)
                queue.put(payload)
            elif console_log:
                formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s")
                ch = logging.StreamHandler(sys.stdout)
                ch.setFormatter(formatter)
                scenario.logger.addHandler(ch)

            if rerun or not scenario.completed:
                if queue is not None:
                    scenario.run(progress_bar=progress_bar, console_log=False)
                else:
                    with logging_redirect_tqdm(loggers=[scenario.logger]):
                        scenario.run(progress_bar=progress_bar, console_log=False)
            else:
                scenario.logger.info("Scenario instance already completed. Skipping...")
        except Exception as e:
            if catch_exceptions:
                trace_output = traceback.format_exc()
                scenario.logger.error(trace_output)
            else:
                raise e
        finally:
            scenario.progress.queue = None
            if queue is not None:
                payload = ScenarioEvent(ScenarioEvent.Type.COMPLETED, id=scenario.id)
                if pickled_queue:
                    payload = pickle.dumps(payload)
                queue.put(payload)
            if qh is not None:
                scenario.logger.removeHandler(qh)
            if ch is not None:
                scenario.logger.removeHandler(ch)
        return scenario

    return _scenario_runner


def get_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(0)
    try:
        # doesn't even have to be reachable
        s.connect(("10.254.254.254", 1))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip


def get_free_port(port: int, max_port: int = 65535) -> int:
    # Refernce: https://stackoverflow.com/a/57086072
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    while port <= max_port:
        try:
            sock.bind(("", port))
            sock.close()
            return port
        except OSError:
            port += 1
    raise IOError("Cannot find a free port.")


DEFAULT_RESULTS_PATH = os.path.join("var", "results")
DEFAULT_RESULTS_SCENARIOS_PATH = os.path.join("var", "results", "scenarios")
DEFAULT_REPORTS_PATH = os.path.join("var", "reports")
ALL_STUDY_PATHS = glob(os.path.join(DEFAULT_RESULTS_PATH, "*"))
DEFAULT_STUDY_PATH = max(ALL_STUDY_PATHS, key=lambda x: os.path.getmtime(x)) if len(ALL_STUDY_PATHS) > 0 else None
DEFAULT_SCENARIO_PATH_FORMAT = "{id}"
DEFAULT_BACKEND = Backend.LOCAL
DEFAULT_EVENTSTREAM_HOST_PORT = 4242
DEFAULT_SLURM_JOBTIME = "24:00:00"
DEFAULT_SLURM_JOBMEMORY = "4G"


class Study:
    def __init__(
        self,
        scenarios: Sequence[Scenario],
        id: Optional[str] = None,
        outpath: str = DEFAULT_RESULTS_PATH,
        scenario_path_format: str = DEFAULT_SCENARIO_PATH_FORMAT,
        logstream: Optional[TextIOBase] = None,
    ) -> None:
        self._scenarios = scenarios
        self._id = id if id is not None else datetime.datetime.now().strftime("Study-%Y-%m-%d-%H-%M-%S")
        self._outpath = outpath
        self._scenario_path_format = scenario_path_format
        self._logstream = logstream if logstream is not None else StringIO()
        self._logger = logging.getLogger(self._id)
        self._logger.setLevel(logging.DEBUG)
        self._verify_scenario_path(scenario_path_format, scenarios)

    @staticmethod
    def _status_monitor(
        queue: Queue,
        logger: Logger,
        study_queue: Optional[Queue] = None,
        pickled: bool = False,
    ) -> None:
        while True:
            record: Optional[Union[logging.LogRecord, Progress.Event]] = None
            payload = queue.get()
            if pickled:
                if payload is not None:
                    record = pickle.loads(payload)
            else:
                record = payload
            if record is None:
                break
            if isinstance(record, Progress.Event):
                Progress.handle(record)
            elif isinstance(record, ScenarioEvent):
                if study_queue is not None:
                    study_queue.put(record)
            else:
                # logger = logging.getLogger(record.name)
                logger.handle(record)
            Progress.refresh()

    def run(
        self,
        catch_exceptions: bool = True,
        progress_bar: bool = True,
        console_log: bool = True,
        backend: Backend = DEFAULT_BACKEND,
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
        queue: Optional[Queue] = None
        process: Optional[Process] = None
        pickled_queue = backend == Backend.SLURM
        if backend == Backend.RAY:
            queue = RayQueue()
        elif backend == Backend.SLURM:
            queue = MultiprocessingQueue()

        pbar = None if not progress_bar else Progress(queue, id=self.id, pickled=pickled_queue)
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
            runner = get_scenario_runner(
                queue, catch_exceptions, progress_bar, console_log, pickled_queue=pickled_queue
            )
            if backend == Backend.RAY:
                monitor = threading.Thread(target=Study._status_monitor, args=(queue, self.logger))
                monitor.start()
                pool = Pool(processes=ray_numprocs, ray_address=ray_address)
                for scenario in pool.imap_unordered(runner, self.scenarios):
                    scenarios.append(scenario)
                    if pbar is not None:
                        pbar.update(1)
                    if eagersave:
                        self.save_scenario(scenario)

            elif backend == Backend.LOCAL:
                for scenario in map(runner, self.scenarios):
                    scenarios.append(scenario)
                    if pbar is not None:
                        pbar.update(1)
                    if eagersave:
                        self.save_scenario(scenario)

            elif backend == Backend.SLURM:

                # If the host port was not specified, we look for the first free port.
                if eventstream_host_port is None:
                    eventstream_host_port = get_free_port(DEFAULT_EVENTSTREAM_HOST_PORT)

                # Set up the process that will host the queue RPC server.
                # Reference: https://stackoverflow.com/a/21146917
                def serve(queue: Queue) -> None:
                    server = zerorpc.Server(queue, heartbeat=45)
                    assert eventstream_host_port is not None
                    server.bind("tcp://0.0.0.0:%d" % eventstream_host_port)

                    def stop_routine():
                        server.stop()

                    def stop_handler(number, frame):
                        gevent.spawn(stop_routine)

                    gevent.signal.signal(signal.SIGTERM, stop_handler)
                    gevent.signal.signal(signal.SIGINT, stop_handler)
                    server.run()
                    sys.stdout.flush()

                process = Process(target=serve, args=(queue,))
                process.start()

                study_queue: Queue = MultiprocessingQueue()
                monitor = threading.Thread(
                    target=Study._status_monitor, args=(queue, self.logger, study_queue, pickled_queue)
                )
                monitor.start()

                if eventstream_host_ip is None:
                    eventstream_host_ip = get_ip()
                address = "tcp://%s:%d" % (eventstream_host_ip, eventstream_host_port)
                self.logger.info("Status event monitor listening on " + address)

                try:

                    # Create a batch job on slurm for every scenario.
                    scenarios_pending = len(self.scenarios)
                    scenarios_running = 0
                    for scenario in self.scenarios:
                        scenarios_pending -= 1
                        if scenario.completed:
                            if pbar is not None:
                                pbar.update(1)
                        else:
                            path = self.save_scenario(scenario)
                            logpath = os.path.join(path, "slurm.log")
                            run_command = "python -m datascope.experiments run-scenario -o %s -e %s" % (path, address)
                            slurm_command = "sbatch --job-name=%s" % self.id
                            slurm_command += " --time=%s" % slurm_jobtime
                            slurm_command += " --mem-per-cpu=%s" % slurm_jobmemory
                            if slurm_constraint is not None:
                                slurm_command += " --constraint=%s" % slurm_constraint
                            if slurm_partition is not None:
                                slurm_command += " --partition=%s" % slurm_partition
                            if slurm_args is not None:
                                slurm_command += " %s " % slurm_args
                            slurm_command += " --output=%s" % logpath
                            slurm_command += ' --wrap="%s"' % run_command
                            result = subprocess.run(slurm_command, capture_output=True, shell=True)
                            if result.returncode != 0:
                                raise RuntimeError(
                                    "Slurm sbatch command gave a non-zero return code. \nstdout:\n%r\nstderr:\n%r\n"
                                    % (result.stdout, result.stderr)
                                )
                            scenarios_running += 1

                        # Check if we need to wait for jobs to finish. We do that either when
                        # the maximum number of allowed jobs has been reached
                        # or if all jobs have been submitted but not all have finished yet.
                        while (slurm_maxjobs is not None and scenarios_running > slurm_maxjobs) or (
                            scenarios_pending <= 0 and scenarios_running > 0
                        ):
                            scenario_event: ScenarioEvent = study_queue.get()
                            if scenario_event.type == ScenarioEvent.Type.COMPLETED:
                                scenarios_running -= 1
                                if pbar is not None:
                                    pbar.update(1)

                except (Exception, KeyboardInterrupt) as e:
                    # Make sure to cancel all submitted slurm jobs in case of an exception or keyboard interrupt.
                    self.logger.info("Running: scancel --name=%s" % self.id)
                    subprocess.run(["scancel", "--name=%s" % self.id])
                    self.logger.info("Slurm jobs canceled.")
                    raise e

                scenarios = Study.load_scenarios(self.path)

            self._scenarios = scenarios

            if pbar is not None:
                pbar.close()

            if backend == Backend.RAY or backend == Backend.SLURM:
                assert queue is not None and monitor is not None
                if process is not None:
                    process.terminate()
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
            raise ValueError("This study has no output path.")
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

    @classmethod
    def load_scenarios(cls, path: str) -> List[Scenario]:
        # Load all scenarios.
        scenarios: List[Scenario] = []
        for attpath in glob(os.path.join(path, "**/attributes.yaml"), recursive=True):
            exppath = os.path.dirname(attpath)
            scenario = Scenario.load(exppath)
            scenarios.append(scenario)
        return scenarios

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
        scenarios = Study.load_scenarios(path)

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
        attributes = sorted(Scenario.attribute_names)
        return Table[Scenario](self._scenarios, attributes, "id")

    @property
    def dataframe(self) -> DataFrame:
        df = pd.concat([scenario.dataframe for scenario in self.scenarios], ignore_index=True)
        df.sort_index(inplace=True)
        return df

    def get_scenarios(self, **attributes: Dict[str, Any]) -> Sequence[Scenario]:
        res: List[Scenario] = []
        for exp in self.scenarios:
            if all(has_attribute_value(exp, name, value) for (name, value) in attributes.items() if hasattr(exp, name)):
                res.append(exp)
        return res


def represent(x: Any):
    if isinstance(x, Enum):
        return repr(x.value)
    else:
        return repr(x)


def stringify(x: Any):
    if isinstance(x, Enum):
        return str(x.value)
    else:
        return str(x)


class Report(ABC):

    reports: Dict[str, Type["Report"]] = {}
    _report: Optional[str] = None

    def __init__(
        self, study: Study, id: Optional[str] = None, groupby: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> None:
        super().__init__()
        self._study = study
        self._id = id if id is not None else "".join(random.choices(string.ascii_lowercase + string.digits, k=10))
        self._groupby: Dict[str, Any] = {} if groupby is None else groupby

    def __init_subclass__(cls: Type["Report"], id: str) -> None:
        cls._report = id
        cls.reports[id] = cls

    @attribute
    def report(self) -> str:
        if self._report is None:
            raise ValueError("Cannot call this on an abstract class instance.")
        return self._report

    @attribute
    def id(self) -> str:
        """A unique identifier of the report."""
        return self._id

    @property
    def groupby(self) -> Dict[str, Any]:
        return self._groupby

    @property
    def study(self) -> Study:
        return self._study

    @classmethod
    def is_valid_config(cls, **attributes: Any) -> bool:
        return True

    @abstractmethod
    def generate(self) -> None:
        raise NotImplementedError()

    def save(
        self,
        path: Optional[str] = None,
        use_groupby: bool = True,
        use_id: bool = False,
        use_subdirs: bool = False,
        saveonly: Optional[Sequence[str]] = None,
    ) -> None:
        if path is None:
            path = os.path.join(self._study.path, "reports")
        basename = "report"
        if use_subdirs:
            if use_groupby:
                groupby = ["%s=%s" % (str(key), str(self._groupby[key])) for key in sorted(self.groupby.keys())]
                path = os.path.join(path, *groupby)
            if use_id:
                path = os.path.join(path, self._id)
            if os.path.splitext(path)[1] != "":
                raise ValueError("The provided path '%s' is not a valid directory path." % path)
        else:
            if use_groupby:
                groupby = [self._groupby[key] for key in sorted(self.groupby.keys())]
                basename = "_".join(
                    [basename]
                    + ["%s=%s" % (str(key), stringify(self._groupby[key])) for key in sorted(self.groupby.keys())]
                )
            if use_id:
                basename = basename + "_id=" + self._id

        os.makedirs(path, exist_ok=True)

        # Save results as separate files.
        props = result.get_properties(type(self))
        results = dict((name, prop.fget(self) if prop.fget is not None else None) for (name, prop) in props.items())
        save_dict(results, path, basename, saveonly=saveonly)

    @classmethod
    def get_instances(
        cls: Type["Report"], study: Study, groupby: Optional[Sequence[str]], **kwargs: Any
    ) -> Iterable["Report"]:
        if cls == Report:
            for id, report in Report.reports.items():
                if kwargs.get("report", None) is None or id in kwargs["report"]:
                    for instance in report.get_instances(study=study, groupby=groupby, **kwargs):
                        yield instance
        else:
            # If grouping attributes were not specified, then we return only a single instance.
            if groupby is None or len(groupby) == 0:
                yield cls(study=study, **kwargs)

            else:
                # Find distinct grouping attribute assignments.
                all_values: List[Tuple] = []
                if len(study.scenarios) > 0:
                    all_values = list(study.dataframe.groupby(groupby).groups.keys())

                for values in all_values:
                    groupby_values = dict((k, v) for (k, v) in zip(groupby, values))
                    yield cls(study=study, groupby=groupby_values, **kwargs)
