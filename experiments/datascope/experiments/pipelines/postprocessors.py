import datascope.importance.common

from typing import Dict, Type, Optional


class Postprocessor(datascope.importance.common.Postprocessor):
    postprocessors: Dict[str, Type["Postprocessor"]] = {}
    summaries: Dict[str, str] = {}
    _postprocessor: Optional[str] = None
    _summary: Optional[str] = None

    def __init_subclass__(
        cls: Type["Postprocessor"],
        abstract: bool = False,
        id: Optional[str] = None,
        summary: Optional[str] = None,
    ) -> None:
        if abstract:
            return

        cls._postprocessor = id if id is not None else cls.__name__
        Postprocessor.postprocessors[cls._postprocessor] = cls
        if summary is not None:
            Postprocessor.summaries[cls._postprocessor] = summary
