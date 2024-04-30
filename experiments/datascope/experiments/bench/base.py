import argparse
import collections.abc
import datetime
import gc
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

from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from glob import glob
from inspect import signature
from io import TextIOBase, StringIO, SEEK_END
from itertools import product
from logging import Logger
from matplotlib.figure import Figure
from methodtools import lru_cache
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
    Generic,
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


class Queue(Protocol):
    def get(self, block: bool = True, timeout: Optional[float] = None) -> Any: ...

    def put(self, item: Any, block: bool = True, timeout: Optional[float] = None) -> None: ...


C = TypeVar("C")


class PropertyTag(property):
    domain: Optional[Iterable] = None
    inherit: bool = False

    @classmethod
    def get_properties(cls: Type[C], target: object) -> Dict[str, C]:
        res: Dict[str, C] = {}
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
        inherit: bool = False,
    ) -> None:
        super().__init__(fget, fset, fdel, doc)
        self.domain = domain
        self.inherit = inherit

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
        return type(self)(fget, fset, fdel, doc, domain=self.domain, inherit=self.inherit)

    __isattribute__ = True


class result(PropertyTag):
    def __init__(
        self,
        fget: Optional[Callable[[Any], Any]] = None,
        fset: Optional[Callable[[Any, Any], None]] = None,
        fdel: Optional[Callable[[Any], None]] = None,
        doc: Optional[str] = None,
        lazy: bool = False,
    ) -> None:
        super().__init__(fget, fset, fdel, doc)
        self.lazy = lazy

    def __call__(
        self,
        fget: Optional[Callable[[Any], Any]] = None,
        fset: Optional[Callable[[Any, Any], None]] = None,
        fdel: Optional[Callable[[Any], None]] = None,
        doc: Optional[str] = None,
    ) -> "result":
        if fget is None:
            fget = self.fget
        if fset is None:
            fset = self.fset
        if fdel is None:
            fdel = self.fdel
        if doc is None:
            doc = self.__doc__
        return type(self)(fget, fset, fdel, doc, lazy=self.lazy)

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
        return [None]
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


def get_property_is_inheritable(target: object, name: str) -> bool:
    member = getattr(target, name)
    if isinstance(member, attribute):
        return member.inherit
    else:
        return False


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


# TODO: Delete this function.
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


T = TypeVar("T")


class LazyLoader(Generic[T]):
    loader: Callable[[], T]
    value: Optional[T]

    def __init__(self, loader: Callable[[], T]) -> None:
        self.loader = loader
        self.value: Optional[T] = None

    def __call__(self) -> T:
        if self.value is None:
            self.value = self.loader()
        return self.value


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


def load_dict(dirpath: str, basename: str, lazy: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    if not os.path.isdir(dirpath):
        raise ValueError("The provided path '%s' does not point to a directory." % dirpath)

    res: Dict[str, Any] = {}
    lazy = [] if lazy is None else lazy

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
            if name in lazy:
                res[name] = LazyLoader(lambda: np.load(path))
            else:
                res[name] = np.load(path)
        elif ext == ".csv":
            if name in lazy:
                res[name] = LazyLoader(lambda: pd.read_csv(path, index_col=0))
            else:
                res[name] = pd.read_csv(path, index_col=0)
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


@dataclass
class AttributeDescriptor:
    attr_type: Optional[type]
    helpstring: Optional[str]
    is_iterable: bool
    domain: Iterable
    default: Any
    inherit: bool = False


def get_all_subclasses(cls):
    all_subclasses = []
    for subclass in cls.__subclasses__():
        all_subclasses.append(subclass)
        all_subclasses.extend(get_all_subclasses(subclass))
    return all_subclasses


def get_class(class_id: str) -> Optional[type]:
    import importlib

    module_name, class_name = class_id.rsplit(".", 1)
    try:
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    except (ModuleNotFoundError, AttributeError):
        return None


class Configurable:
    NESTING_SEPARATOR = "_"
    _class_id: str
    _class_longname: Optional[str] = None
    _class_argname: str = "class"
    _class_abstract: bool = True

    def __init_subclass__(
        cls: Type["Configurable"],
        id: Optional[str] = None,
        longname: Optional[str] = None,
        argname: Optional[str] = None,
        abstract: bool = False,
    ) -> None:
        cls._class_id = id if id is not None else "%s.%s" % (cls.__module__, cls.__name__)
        cls._class_longname = longname if longname is not None else cls.__name__
        if argname is not None:
            cls._class_argname = argname
        cls._class_abstract = abstract

    @lru_cache(maxsize=2)
    @classmethod
    def get_subclasses(cls: Type["Configurable"], include_abstract: bool = False) -> Dict[str, Type["Configurable"]]:
        subclasses = {c.get_class_identifier(): c for c in cls.__subclasses__()}
        subsubclasses: Dict[str, Type["Configurable"]] = {}
        for c in subclasses.values():
            subsubclasses.update(**c.get_subclasses())
        subclasses.update(subsubclasses)
        if include_abstract:
            return subclasses
        else:
            return {k: v for k, v in subclasses.items() if not v._class_abstract}

    @classmethod
    def get_class_identifier(cls: Type["Configurable"]) -> str:
        return cls._class_id

    @lru_cache(maxsize=8)
    @classmethod
    def _get_attribute_descriptors(
        cls: Type["Configurable"],
        flattened: bool = False,
        include_subclasses: bool = False,
        include_classarg: bool = False,
    ) -> Dict[str, AttributeDescriptor]:
        attribute_descriptors = {}
        if include_classarg:
            attribute_descriptors[cls._class_argname] = AttributeDescriptor(
                attr_type=str,
                helpstring="Identifier of the specific %s instance." % cls.__name__.lower(),
                is_iterable=False,
                domain=list(cls.get_subclasses().keys()),
                default=None,
                inherit=False,
            )
        properties = attribute.get_properties(cls)
        for name, prop in properties.items():
            attr_type = get_property_type(cls, name)
            helpstring = get_property_helpstring(cls, name)
            is_iterable = get_property_isiterable(cls, name)
            inherit = get_property_is_inheritable(cls, name)
            default = get_property_default(cls, name)
            if attr_type is not None and issubclass(attr_type, Configurable) and flattened:
                domain = list(attr_type.get_subclasses().keys())
                attribute_descriptors[name] = AttributeDescriptor(
                    attr_type=str,
                    helpstring=helpstring,
                    is_iterable=is_iterable,
                    domain=domain,
                    default=default,
                    inherit=inherit,
                )
                nested_attribute_descriptors = attr_type._get_attribute_descriptors(
                    flattened=True, include_subclasses=True
                )
                for nested_name, nested_descriptor in nested_attribute_descriptors.items():
                    attribute_descriptors[name + cls.NESTING_SEPARATOR + nested_name] = nested_descriptor
            else:
                domain = get_property_domain(cls, name)
                attribute_descriptors[name] = AttributeDescriptor(
                    attr_type=attr_type,
                    helpstring=helpstring,
                    is_iterable=is_iterable,
                    domain=domain,
                    default=default,
                    inherit=inherit,
                )

        if include_subclasses:
            for subclass in cls.get_subclasses().values():
                subclass_descriptors = subclass._get_attribute_descriptors(
                    flattened=flattened, include_subclasses=False
                )
                attribute_descriptors.update(subclass_descriptors)

        return attribute_descriptors

    @cached_property
    def attributes(self) -> Dict[str, Any]:
        props = attribute.get_properties(type(self))
        attributes = dict((name, get_property_value(self, name)) for name in props.keys())
        attributes[self._class_argname] = self.get_class_identifier()
        return attributes

    def __repr__(self) -> str:
        full_class_id = "%s.%s" % (type(self).__module__, type(self).__name__)
        class_id = self.get_class_identifier()
        result = str(self)
        if result.startswith(class_id) and class_id != full_class_id:
            result = result.replace(class_id, full_class_id, 1)
        return result

    def __str__(self) -> str:
        result = self.get_class_identifier()
        attributes = self.attributes
        attribute_string = ", ".join(
            ["%s=%s" % (str(k), stringify(attributes[k])) for k in sorted(attributes) if k != self._class_argname]
        )
        if len(attribute_string) > 0:
            result += "(%s)" % attribute_string
        return result

    @classmethod
    def _compose_attributes(cls: Type["Configurable"], attributes: Dict[str, Any]) -> Dict[str, Any]:
        attribute_descriptors: Dict[str, AttributeDescriptor] = cls._get_attribute_descriptors()
        result: Dict[str, Any] = {}
        skip_prefixes: Set[str] = set()
        for k in sorted(attributes.keys()):
            if any(k.startswith(prefix) for prefix in skip_prefixes):
                continue
            v = attributes[k]
            result[k] = v
            if isinstance(v, str) or isinstance(v, dict):
                target_base_descriptor = attribute_descriptors.get(k, None)
                if target_base_descriptor is not None:
                    target_base_cls = target_base_descriptor.attr_type
                    if target_base_cls is not None and issubclass(target_base_cls, Configurable):
                        target_cls_id = v if isinstance(v, str) else v.get(target_base_cls._class_argname, None)
                        if target_cls_id is None:
                            raise ValueError(
                                "If the key '%s' contains a nested dictionary, it must have a "
                                "key '%s' corresponding to the class identifier." % (k, target_base_cls._class_argname)
                            )

                        target_cls = target_base_cls.get_subclasses().get(target_cls_id, None)
                        if target_cls is None:
                            target_cls = get_class(target_cls_id)
                            if target_cls is not None:
                                if not issubclass(target_cls, target_base_cls) or not issubclass(
                                    target_cls, Configurable
                                ):
                                    target_cls = None

                        if target_cls is not None:
                            target_attributes = {}

                            # If any attributes are inheritable, pass them down.
                            target_attribute_descriptors: Dict[str, AttributeDescriptor] = (
                                target_cls._get_attribute_descriptors()
                            )
                            for kk, vd in target_attribute_descriptors.items():
                                if vd.inherit:
                                    target_attributes[kk] = attributes[kk]

                            # Pass down all attributes that are prefixed with the current key.
                            target_attributes.update(
                                {
                                    kk.removeprefix(k + cls.NESTING_SEPARATOR): vv
                                    for kk, vv in attributes.items()
                                    if kk.startswith(k + cls.NESTING_SEPARATOR)
                                }
                            )

                            # If the value is a nested dictionary, then we reuse its attributes.
                            if isinstance(v, dict):
                                target_attributes.update(
                                    {kk: vv for kk, vv in v.items() if kk != target_base_cls._class_argname}
                                )

                            # Recursively compose target attributes and instantiate the configurable object.
                            target_attributes = target_cls._compose_attributes(target_attributes)
                            result[k] = target_cls(**target_attributes)

                            # Ensure that the nested attributes are not passed to the parent object constructor.
                            skip_prefixes.add(k + cls.NESTING_SEPARATOR)
        return result

    @classmethod
    def _flatten_attributes(cls: Type["Configurable"], attributes: Dict[str, Any]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for k, v in attributes.items():
            if isinstance(v, Configurable):
                result[k] = v.get_class_identifier()
                ejected_attributes = v._flatten_attributes(v.attributes)
                for kk, vv in ejected_attributes.items():
                    result[k + cls.NESTING_SEPARATOR + kk] = vv
            else:
                result[k] = v
        return result

    @classmethod
    def from_dict(cls: Type["Configurable"], attributes: Dict[str, Any]) -> "Configurable":
        class_id = attributes.get(cls._class_argname, None)
        target_class = cls
        if class_id is not None and class_id != cls.get_class_identifier():
            # If class_id is provided, then we need to find the corresponding subclass.
            target_class = cls.get_subclasses().get(class_id, None)
            if target_class is None:
                target_cls = get_class(class_id)
                if target_cls is not None:
                    if not issubclass(target_cls, Configurable):
                        target_cls = None
            if target_class is None:
                raise ValueError("The provided class ID '%s' does not correspond to any valid class." % class_id)
        elif cls._class_abstract:
            # If class_id is not provided, then we need to ensure that cls is not abstract.
            raise ValueError(
                "Function called on abstract class '%s' but argument '%s' specifying subclass is missing."
                % (cls._class_id, cls._class_argname)
            )
        attributes = target_class._compose_attributes(attributes)
        return target_class(**attributes)

    @classmethod
    def add_dynamic_arguments(
        cls: Type["Configurable"],
        parser: argparse.ArgumentParser,
        all_iterable: bool = False,
        single_instance: bool = False,
    ) -> None:
        attribute_descriptors: Dict[str, AttributeDescriptor] = cls._get_attribute_descriptors(
            flattened=True, include_subclasses=True, include_classarg=True
        )
        for name, descriptor in attribute_descriptors.items():
            attribute_domain: Optional[List] = [x.value if isinstance(x, Enum) else x for x in descriptor.domain]
            if attribute_domain == [None]:
                attribute_domain = None
            helpstring = descriptor.helpstring or ("Scenario " + name + ".")
            if descriptor.default is None:
                helpstring += " Default: [all]" if not single_instance else ""
            else:
                helpstring += " Default: %s" % str(descriptor.default)
            nargs = (1 if single_instance else "+") if (descriptor.is_iterable or all_iterable) else None
            parser.add_argument(
                "--%s" % name.replace("_", "-"),
                help=helpstring,
                type=make_type_parser(descriptor.attr_type),
                choices=attribute_domain,
                nargs=nargs,  # type: ignore
            )


class Scenario(Configurable, argname="scenario"):

    def __init__(self, id: Optional[str] = None, logstream: Optional[TextIOBase] = None, **kwargs: Any) -> None:
        super().__init__()
        self._id = id if id is not None else "".join(random.choices(string.ascii_lowercase + string.digits, k=10))
        self._logstream = logstream if logstream is not None else StringIO()
        self._progress = Progress(id=self._id)
        self._attributes: Optional[Dict[str, Any]] = None

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

    @lru_cache(maxsize=1)
    @classmethod
    def get_keyword_replacements(cls: Type["Scenario"]) -> Dict[str, str]:
        return {}

    @classmethod
    def get_instances(cls: Type["Scenario"], **kwargs: Any) -> Iterable["Scenario"]:
        if cls == Scenario:
            # for id, scenario in Scenario.scenarios.items():
            for id, subclass in Scenario.get_subclasses().items():
                if kwargs.get("scenario", None) is None or id in kwargs["scenario"]:
                    assert issubclass(subclass, Scenario)
                    for instance in subclass.get_instances(**kwargs):
                        yield instance
        else:
            attribute_descriptors: Dict[str, AttributeDescriptor] = cls._get_attribute_descriptors(flattened=True)
            domains = []
            names = list(attribute_descriptors.keys())
            for name in names:
                if name in kwargs and kwargs[name] is not None:
                    domain = kwargs[name]
                    if not isinstance(domain, Iterable) or isinstance(domain, str):
                        domain = [domain]
                    domains.append(list(domain))
                else:
                    domains.append(list(attribute_descriptors[name].domain))
            print("len(list(product(*domains)))", len(list(product(*domains))))
            for values in product(*domains):
                attributes = dict((name, value) for (name, value) in zip(names, values) if value is not None)
                composed_attributes = cls._compose_attributes(attributes)
                if cls.is_valid_config(**composed_attributes):
                    scenario = cls(**composed_attributes)
                    # assert isinstance(scenario, Scenario)
                    yield scenario

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
        attributes = self._flatten_attributes(self.attributes)
        save_dict(attributes, path, "attributes")

        # Save results as separate files.
        props: Dict[str, result] = result.get_properties(type(self))
        results = dict((name, prop.fget(self) if prop.fget is not None else None) for (name, prop) in props.items())
        save_dict(results, path, "results")

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
        attributes = cls._compose_attributes(attributes)
        lazy = [k for k, v in result.get_properties(cls).items() if v.lazy]
        results = load_dict(path, "results", lazy=lazy)
        kwargs = {"logstream": logstream}
        scenario = cls.from_dict({**attributes, **results, **kwargs})
        assert isinstance(scenario, Scenario)
        return scenario

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
    job_memory: Optional[str] = None,
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

            # Quickly consume a 70% of the given amount of memory for 10 seconds.
            if job_memory is not None:
                suffixes = {"G": 1024**3, "M": 1024**2, "K": 1024}
                size = int(float(job_memory[:-1]) * suffixes[job_memory[-1]] * np.random.uniform(0.55, 0.7))
                data = bytearray(size)  # noqa: F841
                time.sleep(10)
                del data
                gc.collect()

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
        **kwargs: Any,
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

            scenarios = []
            runner = get_scenario_runner(
                queue, catch_exceptions, progress_bar, console_log, pickled_queue=pickled_queue
            )
            if backend == Backend.LOCAL:
                for scenario in map(runner, self.scenarios):
                    scenarios.append(scenario)
                    if pbar is not None:
                        pbar.update(1)
                    if eagersave:
                        self.save_scenario(scenario)

            elif backend == Backend.RAY:
                monitor = threading.Thread(target=Study._status_monitor, args=(queue, self.logger))
                monitor.start()
                pool = Pool(processes=ray_numprocs, ray_address=ray_address)
                for scenario in pool.imap_unordered(runner, self.scenarios):
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
                            run_command = "python -m datascope.experiments run-scenario -o %s -e %s -m %s" % (
                                path,
                                address,
                                slurm_jobmemory,
                            )
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

        # Save all scenarios.
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
        attributes = sorted(Scenario._get_attribute_descriptors().keys())
        return Table[Scenario](self._scenarios, attributes, "id")

    def get_keyword_replacements(self) -> Dict[str, str]:
        scenario_types: Set[Type[Scenario]] = set([type(scenario) for scenario in self.scenarios])
        result: Dict[str, str] = {}
        for scenario_type in scenario_types:
            result.update(scenario_type.get_keyword_replacements())
        return result

    @property
    def dataframe(self) -> DataFrame:
        df = pd.concat(
            [s.dataframe.assign(scenario=s.get_class_identifier(), id=s.id) for s in self.scenarios], ignore_index=True
        )
        df.sort_index(inplace=True)
        return df

    def get_scenarios(self, **attributes: Dict[str, Any]) -> Sequence[Scenario]:
        res: List[Scenario] = []
        for exp in self.scenarios:
            if all(has_attribute_value(exp, name, value) for (name, value) in attributes.items() if hasattr(exp, name)):
                res.append(exp)
        return res

    @classmethod
    def union(
        cls: Type["Study"],
        *studies: "Study",
        id: Optional[str] = None,
        outpath: str = DEFAULT_RESULTS_PATH,
        scenario_path_format: str = DEFAULT_SCENARIO_PATH_FORMAT,
        logstream: Optional[TextIOBase] = None,
    ) -> "Study":
        scenarios: List[Scenario] = []
        for study in studies:
            scenarios.extend(study.scenarios)
        return Study(scenarios, id=id, outpath=outpath, scenario_path_format=scenario_path_format, logstream=logstream)

    def __add__(self, other: "Study") -> "Study":
        return Study.union(self, other)

    def __radd__(self, other: "Study") -> "Study":
        return Study.union(other, self)


class Report(Configurable):

    def __init__(
        self,
        dataframe: DataFrame,
        id: Optional[str] = None,
        study: Optional[Study] = None,
        partvals: Optional[Dict[str, Any]] = None,
        keyword_replacements: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self._dataframe = dataframe
        self._id = id if id is not None else "".join(random.choices(string.ascii_lowercase + string.digits, k=10))
        self._study = study
        self._partvals: Dict[str, Any] = {} if partvals is None else partvals
        self._keyword_replacements: Dict[str, str] = {} if keyword_replacements is None else keyword_replacements

    @attribute
    def id(self) -> str:
        """A unique identifier of the report."""
        return self._id

    @property
    def partvals(self) -> Dict[str, Any]:
        return self._partvals

    @property
    def study(self) -> Optional[Study]:
        return self._study

    @property
    def dataframe(self) -> DataFrame:
        return self._dataframe

    @property
    def keyword_replacements(self) -> Dict[str, str]:
        return self._keyword_replacements

    @classmethod
    def is_valid_config(cls, **attributes: Any) -> bool:
        return True

    @abstractmethod
    def generate(self) -> None:
        raise NotImplementedError()

    def save(
        self,
        path: Optional[str] = None,
        use_partvals: bool = True,
        use_id: bool = False,
        use_subdirs: bool = False,
        saveonly: Optional[Sequence[str]] = None,
    ) -> None:
        if path is None:
            if self._study is not None:
                path = os.path.join(self._study.path, "reports")
            else:
                path = os.path.join(DEFAULT_REPORTS_PATH, self._id)
        basename = "report"
        if use_subdirs:
            if use_partvals:
                partvals = ["%s=%s" % (str(key), str(self._partvals[key])) for key in sorted(self.partvals.keys())]
                path = os.path.join(path, *partvals)
            if use_id:
                path = os.path.join(path, self._id)
            if os.path.splitext(path)[1] != "":
                raise ValueError("The provided path '%s' is not a valid directory path." % path)
        else:
            if use_partvals:
                partvals = [self._partvals[key] for key in sorted(self.partvals.keys())]
                basename = "_".join(
                    [basename]
                    + ["%s=%s" % (str(key), stringify(self._partvals[key])) for key in sorted(self.partvals.keys())]
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
        cls: Type["Report"],
        study: Study,
        partby: Optional[Sequence[str]],
        keyword_replacements: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> Iterable["Report"]:
        keyword_replacements = {} if keyword_replacements is None else keyword_replacements
        keyword_replacements = {
            **keyword_replacements,
            **study.get_keyword_replacements(),
        }

        if cls == Report:
            for id, report_class in Report.get_subclasses().items():
                if kwargs.get("report", None) is None or id in kwargs["report"]:
                    for instance in report_class.get_instances(
                        study=study, partby=partby, keyword_replacements=keyword_replacements, **kwargs
                    ):
                        yield instance
        else:
            # If grouping attributes were not specified, then we return only a single instance.
            if partby is None or len(partby) == 0:
                yield cls(study=study, dataframe=study.dataframe, **kwargs)

            else:
                # Find distinct grouping attribute assignments.
                all_values: List[Tuple] = []
                partitioned = study.dataframe.groupby(partby)  # type: ignore
                if len(study.scenarios) > 0:
                    all_values = list(partitioned.groups.keys())

                for values in all_values:
                    dataframe = partitioned.get_group(values)
                    partvals = dict((k, v) for (k, v) in zip(partby, values))

                    yield cls(
                        study=study,
                        dataframe=dataframe,
                        partvals=partvals,
                        keyword_replacements=keyword_replacements,
                        **kwargs,
                    )
