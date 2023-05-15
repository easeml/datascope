import copy
import numpy as np

from itertools import product
from numpy.typing import NDArray
from operator import index
from typing import Tuple, Dict, List, Type, Union, Optional, Iterable, Sequence


class AValue:
    maxvalue: Tuple[int, ...] = ()
    infvalue: NDArray
    zerovalue: NDArray
    domainsize: int

    def __init__(self, value: Union[int, Tuple[int, ...], NDArray, None], *args: int) -> None:
        if self.maxvalue == ():
            raise ValueError("Cannot initialize an abstract %s instance." % self.__class__.__name__)
        if len(args) > 0:
            if isinstance(value, int):
                self._value = np.array((value,) + args)
            else:
                raise ValueError("When initializing with multiple arguments, all of them need to be integers.")
        elif isinstance(value, np.ndarray):
            self._value = value
        elif isinstance(value, tuple):
            self._value = np.array(value)
        elif isinstance(value, int):
            self._value = np.array([value])
        elif value is None:
            self._value = self.infvalue
        if self._value.shape != self.zerovalue.shape:
            if self._value.shape == (1,):
                self._value = np.broadcast_to(self._value, self.zerovalue.shape)
            else:
                raise ValueError(
                    "The provided shape is %s but expected either %s or (1,)."
                    % (str(self._value.shape), str(self.zerovalue.shape))
                )
        self._value = self._clip(self._value)

    def _clip(self, value: NDArray) -> NDArray:
        return self.infvalue if (np.any(value < 0) or np.any(value > self.maxvalue)) else value

    def __class_getitem__(cls, key: Tuple[int, ...]) -> Type["AValue"]:
        maxvalue = key
        infvalue = np.array(key) + 1
        zerovalue = np.zeros_like(infvalue)
        domainsize = np.prod(np.array(maxvalue) + 1) + 1
        result = type(
            cls.__name__,
            (cls,),
            dict(maxvalue=maxvalue, infvalue=infvalue, zerovalue=zerovalue, domainsize=domainsize),
        )
        return result

    @classmethod
    def domain(cls) -> Iterable["AValue"]:
        for x in product(*[range(x + 1) for x in cls.maxvalue]):
            yield cls(x)
        yield cls(None)

    @classmethod
    def get_zero(cls) -> "AValue":
        return cls(0)

    @classmethod
    def get_inf(cls) -> "AValue":
        return cls(None)

    @classmethod
    def get_basis(cls, position: int) -> "AValue":
        value = np.zeros_like(cls.zerovalue)
        value[position] = 1
        return cls(value)

    @property
    def value(self) -> Optional[Tuple[int, ...]]:
        if np.array_equal(self._value, self.infvalue):
            return None
        else:
            return tuple(self._value)

    @property
    def is_zero(self) -> bool:
        return np.array_equal(self._value, self.zerovalue)

    @property
    def is_inf(self) -> bool:
        return np.array_equal(self._value, self.infvalue)

    def __repr__(self) -> str:
        return "%s[%s]%s" % (self.__class__.__name__, ", ".join(str(x) for x in self.maxvalue), str(self))

    def __str__(self) -> str:
        if self.is_inf:
            return "(None)"
        elif self.is_zero:
            return "(0)"
        else:
            return "(%s)" % ", ".join(str(x) for x in self._value.ravel().tolist())

    def __hash__(self) -> int:
        return hash(self.value)

    def __bool__(self) -> bool:
        return not np.array_equal(self._value, self.infvalue)

    def __index__(self) -> int:
        if np.array_equal(self._value, self.infvalue):
            return int(self.domainsize - 1)
        m = np.array(self.maxvalue) + 1
        N = m.shape[0]
        v = self._value
        r = sum(v[i] * (np.prod(m[i + 1 :])) for i in range(N))  # noqa: E203
        return int(r)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, AValue):
            return self.value == other.value
        elif isinstance(other, tuple) or isinstance(other, int) or other is None:
            return self.value == type(self)(other).value
        else:
            raise ValueError(
                "Equality not defined between types %s and %s." % (self.__class__.__name__, str(type(other)))
            )

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __add__(self, other: Union["AValue", int, Tuple[int, ...], NDArray, None]) -> "AValue":
        other = type(self)(other) if not isinstance(other, AValue) else other
        return type(self)(self._clip(self._value + other._value))

    def __sub__(self, other: Union["AValue", int, Tuple[int, ...], NDArray, None]) -> "AValue":
        other = type(self)(other) if not isinstance(other, AValue) else other
        return type(self)(self._clip(self._value - other._value))

    def __mul__(self, other: Union["AValue", int, Tuple[int, ...], NDArray, None]) -> "AValue":
        other = type(self)(other) if not isinstance(other, AValue) else other
        return type(self)(self._clip(self._value * other._value))

    def __truediv__(self, other: Union["AValue", int, Tuple[int, ...], NDArray, None]) -> "AValue":
        other = type(self)(other) if not isinstance(other, AValue) else other
        if other.is_zero:
            return type(self)(None)
        return type(self)(self._clip(self._value / other._value))

    def __radd__(self, other: Union["AValue", int, Tuple[int, ...], NDArray, None]) -> "AValue":
        return self.__add__(other)

    def __rsub__(self, other: Union["AValue", int, Tuple[int, ...], NDArray, None]) -> "AValue":
        other = type(self)(other) if not isinstance(other, AValue) else other
        return type(self)(self._clip(other._value - self._value))

    def __rmul__(self, other: Union["AValue", int, Tuple[int, ...], NDArray, None]) -> "AValue":
        return self.__mul__(other)

    def __rtruediv__(self, other: Union["AValue", int, Tuple[int, ...], NDArray, None]) -> "AValue":
        other = type(self)(other) if not isinstance(other, AValue) else other
        return type(self)(self._clip(other._value / self._value))

    def __iadd__(self, other: Union["AValue", int, Tuple[int, ...], NDArray, None]) -> "AValue":
        if not isinstance(other, AValue):
            other = type(self)(other)
        self._value = self._clip(self._value + other._value)
        return self

    def __isub__(self, other: Union["AValue", int, Tuple[int, ...], NDArray, None]) -> "AValue":
        if not isinstance(other, AValue):
            other = type(self)(other)
        self._value = self._clip(self._value - other._value)
        return self

    def __imul__(self, other: Union["AValue", int, Tuple[int, ...], NDArray, None]) -> "AValue":
        if not isinstance(other, AValue):
            other = type(self)(other)
        self._value = self._clip(self._value * other._value)
        return self

    def __itruediv__(self, other: Union["AValue", int, Tuple[int, ...], NDArray, None]) -> "AValue":
        if not isinstance(other, AValue):
            other = type(self)(other)
        if other.is_zero:
            self._value = self.infvalue
        else:
            self._value = self._clip(self._value / other._value)
        return self


class ADD:
    def __init__(
        self,
        units: Union[Sequence[int], int],
        num_candidates: int = 2,
        diameter: int = 1,
        atype: Optional[Type[AValue]] = None,
    ) -> None:
        self.units = list(range(units)) if isinstance(units, int) else list(units)  # type: ignore
        self._units_index = dict((unit, idx) for (idx, unit) in enumerate(self.units))
        self.num_units = len(self.units)
        self.num_candidates = num_candidates
        self.diameter = diameter
        self.atype = atype if atype is not None else AValue[len(self.units)]  # type: ignore
        self.root = 0
        self.nodes = np.zeros((len(self.units), diameter), dtype=int)
        self.child = np.zeros((len(self.units), diameter, self.num_candidates), dtype=int)

        self.adder = np.reshape(
            np.array(
                [copy.deepcopy(self.atype(0)) for _ in range(len(self.units) * diameter * self.num_candidates)],
                dtype=self.atype,
            ),
            newshape=(len(self.units), diameter, self.num_candidates),
        )
        # self.adder = np.full(
        #     (len(self.units), diameter, self.num_candidates), fill_value=self.atype(0), dtype=self.atype
        # )

    def __call__(self, *args: Union[int, bool]) -> AValue:
        if len(args) != len(self.units):
            raise ValueError("Exactly %d arguments expected but %d provided." % (len(self.units), len(args)))

        j = self.root
        result = self.atype.get_zero()
        for i, arg in enumerate(args):
            arg = int(arg)
            result += self.adder[i, j, arg]
            j = self.child[i, j, arg]

        return result

    def restrict(self, unit: int, value: Union[int, bool], inplace: bool = False) -> "ADD":
        value = int(value)
        result = self if inplace else copy.deepcopy(self)
        idx = self._units_index[unit]
        result.units.pop(idx)
        del result._units_index[unit]
        if idx > 0:
            pidx = idx - 1
            for i in range(self.diameter):
                if result.nodes[pidx, i] == 1:
                    for c in range(self.num_candidates):
                        result.adder[pidx, i, c] += self.adder[idx, self.child[pidx, i, c], value]
                        result.child[pidx, i, c] = self.child[idx, self.child[pidx, i, c], value]
        else:
            result.root = self.child[idx, self.root, value]
            for c in range(self.num_candidates):
                result.adder[idx + 1, result.root, c] += self.adder[idx, self.root, value]
            # TODO: Handle root node adders.
        # TODO: Prune dead nodes.
        result.nodes = np.delete(result.nodes, idx, axis=0)
        result.child = np.delete(result.child, idx, axis=0)
        result.adder = np.delete(result.adder, idx, axis=0)
        return result

    def get_update_location(self, units: Iterable[int], values: Iterable[int]) -> List[Tuple]:
        assignment = sorted(zip(units, values), key=lambda x: self._units_index[x[0]])
        if len(assignment) == 0:
            raise ValueError("At one value assignment must be provided.")
        elif len(assignment) == 1:
            unit, value = assignment[0]
            cur_unit_idx = self._units_index[unit]
            cur_nodes = set(self.nodes[cur_unit_idx].nonzero()[0].tolist())
        elif len(assignment) > 1:
            cur_unit_idx = 0
            cur_nodes = {self.root}

        cur_location: List[Tuple] = []
        for unit, value in assignment:
            while self.units[cur_unit_idx] != unit:
                cur_nodes = set(self.child[cur_unit_idx, list(cur_nodes)].flatten())
                cur_unit_idx += 1
            cur_location = [(cur_unit_idx, node, value) for node in cur_nodes]
            cur_nodes = set(self.child[cur_unit_idx, list(cur_nodes), value].flatten())
            cur_unit_idx += 1
        return cur_location

    def update(self, location: List[Tuple], avalue: AValue, increment: bool = False) -> None:
        if increment:
            self.adder[tuple(zip(*location))] += avalue
        else:
            self.adder[tuple(zip(*location))] = np.array([copy.deepcopy(avalue) for _ in range(len(location))])

    def sum(self, other: "ADD") -> "ADD":
        assert self.units == other.units
        assert self.num_candidates == other.num_candidates
        result = ADD(self.units, self.num_candidates, self.diameter * other.diameter, self.atype)
        pnodes: Dict[Tuple[int, int], int] = {}
        cnodes: Dict[Tuple[int, int], int] = {(self.root, other.root): 0}
        for idx in range(len(self.units)):
            pnodes = cnodes
            cnodes = {}
            for (i, j), k in pnodes.items():
                for c in range(self.num_candidates):
                    cidx = cnodes.setdefault((self.child[idx, i, c], other.child[idx, j, c]), len(cnodes))
                    result.nodes[idx, k] = 1
                    result.child[idx, k, c] = cidx
                    result.adder[idx, k, c] = self.adder[idx, i, c] + other.adder[idx, j, c]
        return result

    def modelcount(self) -> NDArray:
        adomain = np.array(list(self.atype.domain()), dtype=self.atype)
        result_current = np.zeros((self.diameter, adomain.shape[0]), dtype=int)
        result_current[:, 0] = 1
        for i in reversed(range(len(self.units))):
            result_previous = result_current
            result_current = np.zeros((self.diameter, adomain.shape[0]), dtype=int)

            for j in range(self.diameter):
                if self.nodes[i, j] != 0:
                    for k, e in enumerate(adomain):
                        if e.is_inf:
                            continue
                        for c in range(self.num_candidates):
                            avalue = e - self.adder[i, j, c]
                            if avalue.is_inf:
                                continue
                            result_current[j, k] += result_previous[self.child[i, j, c], index(avalue)]

        result = result_current[self.root]
        result[-1] = 2 ** len(self.units) - sum(result)  # Correct the count of invalid values.
        return result

    @classmethod
    def construct_tree(cls, units: List[int], num_candidates: int = 2, atype: Optional[Type[AValue]] = None) -> "ADD":
        diameter = num_candidates ** (len(units) - 1)
        d = ADD(units=units, num_candidates=num_candidates, diameter=diameter, atype=atype)
        for i in range(len(units) - 1):
            d.nodes[i, : num_candidates**i] = np.ones(num_candidates**i, dtype=int)
            for c in range(num_candidates):
                d.child[i, : num_candidates**i, c] = np.arange(c, num_candidates ** (i + 1) + c, 2, dtype=int)
        d.nodes[len(units) - 1] = 1
        return d

    @classmethod
    def construct_chain(cls, units: List[int], num_candidates: int = 2, atype: Optional[Type[AValue]] = None) -> "ADD":
        d = ADD(units=units, num_candidates=num_candidates, diameter=1, atype=atype)
        d.nodes = np.ones((len(units), 1), dtype=int)
        return d

    @classmethod
    def concatenate(cls, elements: List["ADD"]) -> "ADD":
        # Prepare the resulting diagram.
        assert all(e.num_candidates == elements[0].num_candidates for e in elements)
        diameter = max(x.diameter for x in elements)
        units: List[int] = sum([x.units for x in elements], [])
        atype = elements[0].atype
        result = ADD(units=units, diameter=diameter, atype=atype)
        result.root = elements[0].root

        # Perform the actual concatenation.
        margins = [diameter - x.diameter for x in elements]
        np.concatenate([np.pad(x.nodes, [(0, 0), (0, margins[i])]) for i, x in enumerate(elements)], out=result.nodes)
        np.concatenate(
            [np.pad(x.child, [(0, 0), (0, margins[i]), (0, 0)]) for i, x in enumerate(elements)], out=result.child
        )
        np.concatenate(
            [
                np.pad(x.adder, [(0, 0), (0, margins[i]), (0, 0)], constant_values=atype(0))  # type: ignore
                for i, x in enumerate(elements)
            ],
            out=result.adder,
        )

        # Reroute child connections between elements.
        idx = -1
        for i in range(len(elements) - 1):
            idx += len(elements[i].units)
            selector = result.nodes[idx].astype("bool")
            result.child[idx, selector, :] = elements[i + 1].root

        return result

    @classmethod
    def stack(cls, factors: List[int], elements: Dict[Tuple[int, ...], "ADD"]) -> "ADD":
        num_candidates = next(iter(elements.values())).num_candidates
        if len(elements) != num_candidates ** len(factors):
            raise ValueError(
                "Given %d factors, the number of elements has to be exactly %d."
                % (len(factors), num_candidates ** (len(factors) + 1))
            )

        # Prepare the resulting diagram.
        diameter = sum(x.diameter for x in elements.values())
        units: List[int] = factors + next(iter(elements.values())).units
        atype = next(iter(elements.values())).atype
        result = ADD(units=units, diameter=diameter, atype=atype)

        # Construct the elements list according to their corresponding factor values.
        elements_list: List["ADD"] = []
        for value in product(*[list(range(num_candidates)) for _ in range(len(factors))]):
            e = elements.get(value, None)
            if e is None:
                raise ValueError("Element for valuation %s not provided." % str(value))
            elements_list.append(e)

        # Construct the header tree made up of factors.
        for i in range(len(factors) - 1):
            result.nodes[i, : num_candidates**i] = np.ones(num_candidates**i, dtype=int)
            for c in range(num_candidates):
                result.child[i, : 2**i, c] = np.arange(c, num_candidates ** (i + 1) + c, 2, dtype=int)
        result.nodes[len(factors) - 1, : num_candidates ** (len(factors) - 1)] = 1

        # Route factor leaves to element roots.
        offsets = np.zeros(len(elements_list), dtype=int)
        np.cumsum([elements_list[i].diameter for i in range(len(elements_list) - 1)], out=offsets[1:])
        roots = np.array([elements_list[i].root for i in range(len(elements_list))], dtype=int) + offsets
        nf = len(factors)
        ne = len(elements)
        for c in range(num_candidates):
            result.child[nf - 1, : num_candidates ** (nf - 1), c] = [
                roots[i] for i in range(ne) if i % num_candidates == c
            ]  # noqa: E203

        # Perform the actual stacking of elements. We need to apply offsets to child node pointers.
        np.concatenate([x.nodes for x in elements_list], axis=1, out=result.nodes[len(factors) :])  # noqa: E203
        np.concatenate(
            [x.child + offsets[i] for i, x in enumerate(elements_list)],
            axis=1,
            out=result.child[len(factors) :],  # noqa: E203
        )
        np.concatenate([x.adder for x in elements_list], axis=1, out=result.adder[len(factors) :])  # noqa: E203

        return result

    def __repr__(self) -> str:
        return "ROOT: %d\n" % self.root + "\n".join(
            "[% 3s]: %s"
            % (
                str(v),
                " ".join(
                    "[%d]" % j
                    + "".join(
                        "(%d, %s)" % (self.child[i, j, k], self.adder[i, j, k]) for k in range(self.num_candidates)
                    )
                    for j, n in enumerate(self.nodes[i])
                    if n != 0
                ),
            )
            for i, v in enumerate(self.units)
        )
