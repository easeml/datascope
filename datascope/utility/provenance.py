from __future__ import annotations

import collections.abc
import itertools
import numpy as np

from abc import ABC, abstractmethod
from copy import deepcopy
from numpy.typing import NDArray
from typing import (
    List,
    Dict,
    Optional,
    Sequence,
    Mapping,
    MutableSequence,
    Iterator,
    Iterable,
    Hashable,
    TypeVar,
    Type,
    overload,
)


def _hashable_prefix_map(target: Hashable, prefix: Hashable) -> Hashable:
    if isinstance(target, str):
        if isinstance(prefix, str):
            return prefix + target
        elif isinstance(prefix, tuple):
            return prefix + (target,)
        else:
            return (prefix, target)
    elif isinstance(target, tuple):
        if isinstance(prefix, tuple):
            return prefix + target
        else:
            return (prefix,) + target
    else:
        if isinstance(prefix, tuple):
            return prefix + (target,)
        else:
            return (prefix, target)


T = TypeVar("T", bound=np.generic)


def _pad_array(array: NDArray[T], shape: tuple) -> NDArray[T]:
    if array.shape == shape:
        return array
    else:
        if len(array.shape) != len(shape):
            raise ValueError(
                "The shape of the provided array has %d dimensions but the provided shape has %d."
                % (len(array.shape), len(shape))
            )
        return np.pad(
            array,
            [(0, d_target - d_array) for (d_array, d_target) in zip(array.shape, shape)],
            mode="constant",
            constant_values=-1,
        )


class Units(Mapping[Hashable, "Units.Unit"]):
    """Represents a set of units."""

    class Unit:
        def __init__(self, key: Hashable, parent: "Units") -> None:
            self._key = key
            self._parent = parent

        @property
        def key(self) -> Hashable:
            return self._key

        @property
        def parent(self) -> Units:
            return self._parent

        def __repr__(self) -> str:
            return "%s[%s]" % (self.parent._name, repr(self._key))

        def __deepcopy__(self, memo):
            return Units.Unit(key=self._key, parent=self._parent)

        def __eq__(self, other: object) -> Equality:  # type: ignore
            if not isinstance(other, Hashable):
                raise ValueError(
                    "An equality predicate must involve a unit on the LHS and a hashable object on the RHS."
                )
            if other not in self._parent._candidates_index:
                if self._parent._candidates_frozen:
                    raise ValueError("The value %s is not found among the candidates of this units set." % repr(other))
                else:
                    self._parent._candidates_index[other] = len(self._parent._candidates)
                    self._parent._candidates.append(other)
            return Equality(self, other)

    def __init__(
        self,
        units: Sequence[Hashable] | int | None = None,
        candidates: Sequence[Hashable] | int | None = None,
        *,
        name: str = "x",
    ) -> None:
        self._units: List[Hashable] = []
        self._units_frozen = units is not None
        if isinstance(units, int):
            self._units = list(range(units))
        elif isinstance(units, Sequence):
            self._units = list(units)
        elif units is not None:
            raise ValueError(
                "The provided units argument must be either a None, an integer or a sequence of hashable objects."
            )

        self._candidates: List[Hashable] = []
        self._candidates_frozen = candidates is not None
        if isinstance(candidates, int):
            self._candidates = list(range(candidates))
        elif isinstance(candidates, Sequence):
            self._candidates = list(candidates)
        elif candidates is not None:
            raise ValueError(
                "The provided candidates argument must be either a None, an integer or a sequence of hashable objects."
            )

        self._units_index = dict((x, i) for i, x in enumerate(self._units))
        self._candidates_index = dict((x, i) for i, x in enumerate(self._candidates))
        self._name = name

    @property
    def units(self) -> Sequence[Hashable]:
        return self._units

    @property
    def units_index(self) -> Dict[Hashable, int]:
        return self._units_index

    @property
    def candidates(self) -> Sequence[Hashable]:
        return self._candidates

    @property
    def candidates_index(self) -> Dict[Hashable, int]:
        return self._candidates_index

    @property
    def name(self) -> str:
        """The name of the unit set."""
        return self._name

    def __getitem__(self, key: Hashable) -> Unit:
        if key not in self._units_index:
            if self._units_frozen:
                raise KeyError("The specified unit does not exist in this unit set.")
            self._units_index[key] = len(self._units)
            self._units.append(key)
        return Units.Unit(key, self)

    def __iter__(self) -> Iterator[Hashable]:
        return iter(self._units)

    def __len__(self) -> int:
        return len(self._units)

    def __contains__(self, key: Hashable) -> bool:
        return key in self._units_index

    def prefix(self, prefix: Hashable, inplace: bool = False) -> "Units":
        target = self if inplace else deepcopy(self)
        target._units = [_hashable_prefix_map(target=u, prefix=prefix) for u in target._units]
        target._units_index = dict(
            (_hashable_prefix_map(target=u, prefix=prefix), i) for (u, i) in target._units_index.items()
        )
        return target

    def union(self, other: Units, prefix: Optional[Hashable] = None, other_prefix: Optional[Hashable] = None) -> Units:
        target = self if prefix is None else self.prefix(prefix)
        other = other if other_prefix is None else other.prefix(other_prefix)
        for key in other._units_index:
            if key not in target._units_index:
                target._units_index[key] = len(target._units)
                target._units.append(key)
        for key in other._candidates_index:
            if key not in target._candidates_index:
                target._candidates_index[key] = len(target._candidates)
                target._candidates.append(key)
        target._units_frozen = self._units_frozen or other._units_frozen
        target._candidates_frozen = self._candidates_frozen or other._candidates_frozen
        return target


class Expression(ABC):
    """An abstract expression defined over a set of units which can be evaluated given a set of value assignments."""

    def __init__(self, units: Units) -> None:
        super().__init__()
        self._units = units

    @abstractmethod
    def eval(self, values: Sequence[Hashable] | Dict[Hashable, Hashable] | NDArray[np.int_]) -> bool:
        pass

    @property
    @abstractmethod
    def data(self) -> NDArray[np.int_]:
        pass

    @property
    def units(self) -> Units:
        return self._units

    @classmethod
    def from_data(cls: Type[Expression], data: NDArray[np.int_], units: Units) -> Expression:
        if data.ndim == 1:
            return Equality.from_data(data=data, units=units)
        elif data.ndim == 2:
            return Conjunction.from_data(data=data, units=units)
        elif data.ndim == 3:
            return Disjunction.from_data(data=data, units=units)
        else:
            raise ValueError("The provided data array has %d dimensions but 1, 2, or 3 were expected." % data.ndim)


class Equality(Expression):
    def __init__(self, unit: Units.Unit, value: Hashable) -> None:
        super().__init__(units=unit.parent)
        self._unit = unit
        self._value = value

    @property
    def unit(self) -> Units.Unit:
        return self._unit

    @property
    def value(self) -> Hashable:
        return self._value

    def eval(self, values: Sequence[Hashable] | Dict[Hashable, Hashable] | NDArray[np.int_]) -> bool:
        if isinstance(values, dict):
            default_candidate = next(iter(self._units.candidates))
            return values.get(self._unit.key, default_candidate) == self.value
        else:
            key = self._unit.parent._units_index[self._unit.key]
            return values[key] == self.value

    def __repr__(self) -> str:
        return "%s == %s" % (repr(self._unit), repr(self._value))

    def __deepcopy__(self, memo: dict) -> Equality:
        return Equality(unit=deepcopy(self._unit, memo), value=self._value)

    @overload
    def __and__(self, other: Equality | Conjunction) -> Conjunction: ...

    @overload
    def __and__(self, other: Disjunction) -> Disjunction: ...

    def __and__(self, other: Expression) -> Expression:
        if isinstance(other, Equality):
            return Conjunction(deepcopy(self), deepcopy(other))
        elif isinstance(other, Conjunction):
            return Conjunction(deepcopy(self), *[deepcopy(e) for e in other._elements])
        elif isinstance(other, Disjunction):
            return Disjunction(*[self & e for e in other._elements])
        else:
            raise ValueError("Unsupported operand type '%s'." % type(other))

    def __or__(self, other: Expression) -> Disjunction:
        if isinstance(other, Equality):
            return Disjunction(Conjunction(deepcopy(self)), Conjunction(deepcopy(other)))
        elif isinstance(other, Conjunction):
            return Disjunction(Conjunction(deepcopy(self)), deepcopy(other))
        elif isinstance(other, Disjunction):
            return Disjunction(Conjunction(deepcopy(self)), *[deepcopy(e) for e in other._elements])
        else:
            raise ValueError("Unsupported operand type '%s'." % type(other))

    @property
    def data(self) -> NDArray[np.int_]:
        units = self._unit.parent
        return np.array([units.units_index[self._unit._key], units.candidates_index[self._value]], dtype=np.int_)

    @classmethod
    def from_data(cls: Type[Expression], data: NDArray[np.int_], units: Units) -> Equality:
        if data.ndim != 1:
            raise ValueError("The provided data array has %d dimensions but 1 was expected." % data.ndim)
        if data.shape[0] != 2:
            raise ValueError("The provided data array has %d elements but 2 were expected." % data.shape[0])
        return Equality(unit=units[units.units[data[0]]], value=units.candidates[data[1]])


class Conjunction(Expression):
    def __init__(self, *elements: Equality) -> None:
        if len(set(id(e.units) for e in elements)) > 1:
            raise ValueError("The provided list of equality expressions must be defined over the same set of units.")
        super().__init__(units=elements[0].units)
        self._elements = list(elements)

    def eval(self, values: Sequence[Hashable] | Dict[Hashable, Hashable] | NDArray[np.int_]) -> bool:
        return all(e.eval(values) for e in self._elements)

    def __repr__(self) -> str:
        return " & ".join("(%s)" % repr(element) for element in self._elements)

    @overload
    def __and__(self, other: Equality | Conjunction) -> Conjunction: ...

    @overload
    def __and__(self, other: Disjunction) -> Disjunction: ...

    def __and__(self, other: Expression) -> Expression:
        if isinstance(other, Equality):
            return Conjunction(*[deepcopy(e) for e in self._elements] + [deepcopy(other)])
        elif isinstance(other, Conjunction):
            return Conjunction(*[deepcopy(e) for e in self._elements] + [deepcopy(e) for e in other._elements])
        elif isinstance(other, Disjunction):
            return Disjunction(*[self & e for e in other._elements])
        else:
            raise ValueError("Unsupported operand type '%s'." % type(other))

    def __or__(self, other: Expression) -> Disjunction:
        if isinstance(other, Equality):
            return Disjunction(deepcopy(self), Conjunction(deepcopy(other)))
        elif isinstance(other, Conjunction):
            return Disjunction(deepcopy(self), deepcopy(other))
        elif isinstance(other, Disjunction):
            return Disjunction(deepcopy(self), *[deepcopy(e) for e in other._elements])
        else:
            raise ValueError("Unsupported operand type '%s'." % type(other))

    @property
    def data(self) -> NDArray[np.int_]:
        return np.stack([element.data for element in self._elements], axis=0, dtype=np.int_)

    @classmethod
    def from_data(cls: Type[Expression], data: NDArray[np.int_], units: Units) -> Conjunction:
        if data.ndim != 2:
            raise ValueError("The provided data array has %d dimensions but 2 was expected." % data.ndim)
        elements = [Equality.from_data(data[i], units) for i in range(data.shape[0]) if not np.any(data[i] == -1)]
        return Conjunction(*elements)


class Disjunction(Expression):
    def __init__(self, *elements: Conjunction) -> None:
        if len(set(id(e.units) for e in elements)) > 1:
            raise ValueError("The provided list of equality expressions must be defined over the same set of units.")
        super().__init__(units=elements[0].units)
        if not all(isinstance(e, Conjunction) for e in elements):
            raise ValueError("The provided elements must be either a conjunction or a disjunction.")
        self._elements: List[Conjunction] = list(elements)

    def eval(self, values: Sequence[Hashable] | Dict[Hashable, Hashable] | NDArray[np.int_]) -> bool:
        return any(e.eval(values) for e in self._elements)

    def __repr__(self) -> str:
        return " | ".join("(%s)" % repr(element) for element in self._elements)

    def __and__(self, other: Expression) -> Disjunction:
        if isinstance(other, Equality) or isinstance(other, Conjunction):
            return Disjunction(*[e & other for e in self._elements])
        elif isinstance(other, Disjunction):
            return Disjunction(*[a & b for a, b in itertools.product(self._elements, other._elements)])
        else:
            raise ValueError("Unsupported operand type '%s'." % type(other))

    def __or__(self, other: Expression) -> Disjunction:
        if isinstance(other, Equality):
            return Disjunction(*[deepcopy(e) for e in self._elements] + [Conjunction(deepcopy(other))])
        elif isinstance(other, Conjunction):
            return Disjunction(*[deepcopy(e) for e in self._elements] + [deepcopy(other)])
        elif isinstance(other, Disjunction):
            return Disjunction(*[deepcopy(e) for e in self._elements] + [deepcopy(e) for e in other._elements])
        else:
            raise ValueError("Unsupported operand type '%s'." % type(other))

    @property
    def data(self) -> NDArray[np.int_]:
        element_data = [element.data for element in self._elements]
        shape = (max(data.shape[0] for data in element_data), 2)
        return np.stack([_pad_array(data, shape) for data in element_data], axis=0, dtype=np.int_)

    @classmethod
    def from_data(cls: Type[Expression], data: NDArray[np.int_], units: Units) -> Disjunction:
        if data.ndim != 3:
            raise ValueError("The provided data array has %d dimensions but 3 was expected." % data.ndim)
        elements = [Conjunction.from_data(data[i], units) for i in range(data.shape[0]) if not np.all(data[i] == -1)]
        return Disjunction(*elements)


class Provenance(MutableSequence[Expression]):
    """Enables efficient construction, storage and querying of provenance information for a set of tuples.

    Provenance information for each tuple is represented as a logical formula over a set of variables which
    we refer to as `units`. To each unit we can assign a value taken from a predefined set of possible values
    which we refer to as `candidates`. Provenance formulas are represented in *disjunctive normal form* [^dnf].
    Specifically, it is represented as a disjunction of a list of conjunctions where each element of
    the conjunction is an expression fomed as `(unit == candiadte)`. For example, if we have
    units `x[1]`, `x[2]` and `x[3]`, and candidate values `0`, `1` and `2`, one possible provenance formula could be:

    ```
    ((x(1) == 0) & (x(2) == 1)) | ((x(1) == 1) & (x(3) == 3))
    ```

    Given a set of `values` where each unit is mapped to a single candiate value, we can evaluate
    a provenance formula and if it evaluates to `True`, then the tuple which is associated with the formula should
    appear in the dataset. For example, the value assignment `x = {1: 1, 2: 2, 3: 3}` would
    evaluate the above formula to `True` and the value assignment `x = {1: 0, 2: 2, 3: 3}` would
    evaluate it to `False`.

    Attributes
    ----------
    units: int or Sequence[int]
        Specifies unit identifiers. If an integer `N` is passed, then we assume there are `N` units with identifiers
        from `0` to `N-1`. If we would like to speficy custom unit identifiers, we can pass a sequence of integers.

    candidates: None or int or Sequence[int]
        Specifies candidate identifiers. If we pass `None` (default), then we assume there are two candidates: 0 and 1.
        If an integer `M` is passed, then we assume each tuple has `M` candidates with identifiers
        going from `0` to `M-1`. If we would like to speficy custom candidate identifiers,
        we can pass a sequence of integers.

    data: None or Sequence[int] or NDArray[np.int_]
        Allows us to initialize the object with provenance data. If we pass `None` (default), provenance data
        is generated by simply assuming that each unit corresponds to a unique tuple and the order of tuples is the
        same as the order of units.


    References
    ----------
    [^dnf]: "Disjunctive normal form", https://en.wikipedia.org/wiki/Disjunctive_normal_form
    """

    def __init__(
        self,
        expressions: Optional[Sequence[Expression]] = None,
        *,
        units: Optional[Units | Sequence[Hashable] | int] = None,
        candidates: Sequence[Hashable] | int = 2,
        data: Optional[Sequence[int] | NDArray[np.int_]] = None,
    ) -> None:
        self._is_simple = False

        if expressions is not None and len(expressions) > 0:
            # If the expressions were passed as arguments, ensure that no other arguments were passed.
            if units is not None or candidates != 2 or data is not None:
                raise ValueError(
                    "Since an explicit list of expressions has been provided, arguments for "
                    "units, candidates and data must be left unspecified."
                )

            # Extract expression units set and ensure it is the same one accross all expressions.
            if len(set(id(e.units) for e in expressions)) > 1:
                raise ValueError(
                    "The provided list of equality expressions must be defined over the same set of units."
                )
            self._units = expressions[0].units

            # Extract expression data and ensure it is 3-dimensional.
            expression_data = list(e.data for e in expressions)
            expression_data = [
                (
                    d
                    if d.ndim == 3
                    else d.reshape([1, d.shape[0], d.shape[1]]) if d.ndim == 2 else d.reshape([1, 1, d.shape[0]])
                )
                for d in expression_data
            ]

            # Pad expression data if needed.
            max_disjunctions = max(d.shape[0] for d in expression_data)
            max_conjunctions = max(d.shape[1] for d in expression_data)
            expression_data = [_pad_array(d, shape=(max_disjunctions, max_conjunctions, 2)) for d in expression_data]

            # Stack expression data to produce the data array.
            self._data: NDArray[np.int_] = np.stack(expression_data, axis=0, dtype=np.int_)

        else:
            # If units were not provided, we will gather all unique units we encounter in the data array.
            if units is None:
                if data is None:
                    raise ValueError("Both units and data are set to None.")
                unit_data = (
                    data
                    if isinstance(data, collections.abc.Sequence) or data.ndim == 1 or data.shape[-1] == 1
                    else data[..., 0]
                )
                units = [unit for unit in np.unique(unit_data) if unit != -1]

            self._units = units if isinstance(units, Units) else Units(units=units, candidates=candidates)
            num_units = len(self._units.units)
            num_candidates = len(self._units.candidates)

            if data is None:
                data = np.arange(num_units, dtype=np.int_)
                self._is_simple = True
            elif isinstance(data, collections.abc.Sequence):
                data = np.array(data, dtype=np.int_)
            elif not np.issubdtype(data.dtype, np.integer):
                raise ValueError("The data must be an integer array.")

            if data.ndim == 1:
                data = np.stack(
                    [np.repeat(data, num_candidates - 1), np.tile(np.arange(1, num_candidates), len(data))], axis=-1
                )
            assert data is not None and isinstance(data, np.ndarray)
            if data.ndim == 2:
                data = data.reshape([-1, 1, 1, 2])
            elif data.ndim == 3:
                data = data.reshape([-1, 1, data.shape[1], 2])
            self._data = data

    @property
    def units(self) -> Sequence[Hashable]:
        return self._units.units

    @property
    def units_index(self) -> Dict[Hashable, int]:
        return self._units.units_index

    @property
    def candidates(self) -> Sequence[Hashable]:
        return self._units.candidates

    @property
    def candidates_index(self) -> Dict[Hashable, int]:
        return self._units.candidates_index

    @property
    def data(self) -> NDArray[np.int_]:
        return self._data

    @property
    def num_tuples(self) -> int:
        return self._data.shape[0]

    @property
    def max_disjunctions(self) -> int:
        return self._data.shape[1]

    @property
    def max_conjunctions(self) -> int:
        return self._data.shape[2]

    @property
    def num_units(self) -> int:
        return len(self._units.units)

    @property
    def num_candidates(self) -> int:
        return len(self._units.candidates)

    @property
    def is_simple(self) -> bool:
        return self._is_simple

    @overload
    def __getitem__(self, index: int) -> Expression: ...

    @overload
    def __getitem__(self, index: slice) -> Provenance: ...

    @overload
    def __getitem__(
        self, index: Sequence[int] | Sequence[bool] | NDArray[np.int_] | NDArray[np.bool_]
    ) -> Provenance: ...

    def __getitem__(
        self, index: int | slice | Sequence[int] | Sequence[bool] | NDArray[np.int_] | NDArray[np.bool_]
    ) -> Expression | "Provenance":
        if isinstance(index, int):
            return Expression.from_data(self._data[index], self._units)
        else:
            return Provenance(units=self._units, data=self._data[index])

    @overload
    def __setitem__(self, index: int, value: Expression) -> None: ...

    @overload
    def __setitem__(self, index: slice, value: Iterable[Expression]) -> None: ...

    @overload
    def __setitem__(self, index: Sequence[int], value: Iterable[Expression]) -> None: ...

    @overload
    def __setitem__(self, index: NDArray, value: Iterable[Expression]) -> None: ...

    def __setitem__(
        self,
        index: int | slice | Sequence[int] | Sequence[bool] | NDArray[np.int_] | NDArray[np.bool_],
        value: Expression | Iterable[Expression],
    ) -> None:
        expression_data = [value.data] if isinstance(value, Expression) else list(v.data for v in value)
        expression_data = [
            (
                d
                if d.ndim == 3
                else d.reshape([1, d.shape[0], d.shape[1]]) if d.ndim == 2 else d.reshape([1, 1, d.shape[0]])
            )
            for d in expression_data
        ]
        max_disjunctions = max(d.shape[0] for d in expression_data)
        max_conjunctions = max(d.shape[1] for d in expression_data)
        self._data = _pad_array(self._data, shape=(self._data.shape[0], max_disjunctions, max_conjunctions, 2))
        if isinstance(index, int):
            self._data[index] = expression_data[0]
        else:
            self._data[index] = np.stack(expression_data, axis=0, dtype=np.int_)

    @overload
    def __delitem__(self, index: int) -> None: ...

    @overload
    def __delitem__(self, index: slice) -> None: ...

    @overload
    def __delitem__(self, index: Sequence[int] | Sequence[bool] | NDArray[np.int_] | NDArray[np.bool_]) -> None: ...

    def __delitem__(
        self, index: int | slice | Sequence[int] | Sequence[bool] | NDArray[np.int_] | NDArray[np.bool_]
    ) -> None:
        self._data = np.delete(self._data, index, axis=0)

    def __len__(self) -> int:
        return self._data.shape[0]

    def insert(self, index: int, value: Expression) -> None:
        self._data = np.insert(self._data, index, -1, axis=0)
        self[index] = value

    def query(
        self, values: Sequence[int] | Dict[Hashable, Hashable] | NDArray[np.int_], dtype: type = bool
    ) -> NDArray[np.int_] | NDArray[np.bool_]:
        if isinstance(values, dict):
            default_candidate = self._units.candidates[0]
            values = [self._units.candidates_index[values.get(x, default_candidate)] for x in self.units]
        if not isinstance(values, np.ndarray):
            values = np.array(values)
        if not values.ndim == 1:
            raise ValueError("The values must have a single dimension.")
        if not values.shape[0] == self.num_units:
            raise ValueError("If values were provided as an array, the size must be the same as the number of units.")
        values = np.append(values, -1)

        result = np.equal(values[self._data[:, :, :, 0]], self._data[:, :, :, 1])
        result = result.squeeze(axis=2) if result.shape[2] == 1 else np.all(result, axis=2)
        result = result.squeeze(axis=1) if result.shape[1] == 1 else np.any(result, axis=1)  # TODO: This is incorrect.
        # Specifically, when some elements of data are -1 then some equalities would be (-1 == -1) which are true
        # and this is fine for all() because True is a netural element. But in any() this situation causes problems.
        if dtype == int:
            result = np.argwhere(result)
        return result

    def fork(self, size: int | Sequence[int] | NDArray[np.int_]) -> Provenance:
        return Provenance(units=self._units, data=self._data.repeat(size, axis=0))

    def join(self, other: Provenance, prefix: Optional[Hashable], other_prefix: Optional[Hashable]) -> Provenance:
        units = self._units.union(other=other._units, prefix=prefix, other_prefix=other_prefix)
        self_data = np.repeat(self._data, len(other), axis=0)
        other_data = np.tile(other._data, (len(self), 1, 1, 1))
        data = np.stack([self_data, other_data], axis=2, dtype=np.int_)
        return Provenance(units=units, data=data)

    def filter(self, selector: Sequence[bool] | NDArray[np.bool_]) -> Provenance:
        return Provenance(units=self._units, data=np.delete(self._data, selector, axis=0))
