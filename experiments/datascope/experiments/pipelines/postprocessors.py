from abc import abstractmethod
from datascope.importance import common
from methodtools import lru_cache
from typing import Dict, Type

from ..bench import Configurable
from ..datasets import Dataset


class Postprocessor(Configurable, abstract=True, argname="postprocessor"):
    """This is a configurable class that can be plugged into a scenario and used to construct a postprocessor."""

    @abstractmethod
    def construct(self: "Postprocessor", dataset: Dataset) -> common.Postprocessor:
        pass

    @lru_cache(maxsize=1)
    @classmethod
    def get_keyword_replacements(cls: Type["Postprocessor"]) -> Dict[str, str]:
        result: Dict[str, str] = {}
        for class_id, class_type in cls.get_subclasses().items():
            assert issubclass(class_type, Postprocessor)
            result[class_id] = class_type._class_longname
        return result


class IdentityPostprocessor(Postprocessor, id="identity"):
    """An pass-through postprocessor that does not perform any transormation on the data."""

    def construct(self: "IdentityPostprocessor", dataset: Dataset) -> common.Postprocessor:
        return common.IdentityPostprocessor()
