from enum import Enum
from sklearn.pipeline import Pipeline
from sklearn.base import ClassifierMixin
from typing import Optional, Union, Callable, Literal, Any


class ImportanceMeasure(Enum):
    SAHPLEY = "shapley"
    INFOGAIN = "infogain"
    INFLUENCE = "influence"


class ImportanceMethod(Enum):
    BRUTEFORCE = "bruteforce"
    MONTECARLO = "montecarlo"
    NEIGHBOR = "neighbor"


class Debugger:
    def __init__(
        self,
        pipeline: Union[Pipeline, ClassifierMixin],
        importance: Literal[ImportanceMeasure.SAHPLEY, ImportanceMeasure.INFOGAIN, ImportanceMeasure.INFLUENCE],
        method: Literal[ImportanceMethod.BRUTEFORCE, ImportanceMethod.MONTECARLO, ImportanceMethod.NEIGHBOR],
        utility: Union[Callable, str],
        model: Optional[ClassifierMixin],
    ) -> None:

        self.pipeline = pipeline
        self.importance = importance
        self.method = method
        self.utility = utility
        self.model = model

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass
