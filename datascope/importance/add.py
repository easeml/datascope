import copy
import numpy as np

from itertools import product
from numpy import ndarray
from operator import index
from typing import Tuple, Dict, List, Type, Union, Optional, Iterable


class AValue:

    maxvalue: Tuple[int, ...] = ()
    infvalue: ndarray
    zerovalue: ndarray
    domainsize: int

    def __init__(self, value: Union[int, Tuple[int, ...], ndarray, None], *args: int) -> None:
        if self.maxvalue == ():
            raise ValueError("Cannot initialize an abstract %s instance." % self.__class__.__name__)
        if len(args) > 0:
            if isinstance(value, int):
                self._value = np.array((value,) + args)
            else:
                raise ValueError("When initializing with multiple arguments, all of them need to be integers.")
        elif isinstance(value, ndarray):
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

    def _clip(self, value: ndarray) -> ndarray:
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

    def __add__(self, other: Union["AValue", int, Tuple[int, ...], ndarray, None]) -> "AValue":
        other = type(self)(other) if not isinstance(other, AValue) else other
        return type(self)(self._clip(self._value + other._value))

    def __sub__(self, other: Union["AValue", int, Tuple[int, ...], ndarray, None]) -> "AValue":
        other = type(self)(other) if not isinstance(other, AValue) else other
        return type(self)(self._clip(self._value - other._value))

    def __mul__(self, other: Union["AValue", int, Tuple[int, ...], ndarray, None]) -> "AValue":
        other = type(self)(other) if not isinstance(other, AValue) else other
        return type(self)(self._clip(self._value * other._value))

    def __truediv__(self, other: Union["AValue", int, Tuple[int, ...], ndarray, None]) -> "AValue":
        other = type(self)(other) if not isinstance(other, AValue) else other
        if other.is_zero:
            return type(self)(None)
        return type(self)(self._clip(self._value / other._value))

    def __radd__(self, other: Union["AValue", int, Tuple[int, ...], ndarray, None]) -> "AValue":
        return self.__add__(other)

    def __rsub__(self, other: Union["AValue", int, Tuple[int, ...], ndarray, None]) -> "AValue":
        other = type(self)(other) if not isinstance(other, AValue) else other
        return type(self)(self._clip(other._value - self._value))

    def __rmul__(self, other: Union["AValue", int, Tuple[int, ...], ndarray, None]) -> "AValue":
        return self.__mul__(other)

    def __rtruediv__(self, other: Union["AValue", int, Tuple[int, ...], ndarray, None]) -> "AValue":
        other = type(self)(other) if not isinstance(other, AValue) else other
        return type(self)(self._clip(other._value / self._value))

    def __iadd__(self, other: Union["AValue", int, Tuple[int, ...], ndarray, None]) -> "AValue":
        if not isinstance(other, AValue):
            other = type(self)(other)
        self._value = self._clip(self._value + other._value)
        return self

    def __isub__(self, other: Union["AValue", int, Tuple[int, ...], ndarray, None]) -> "AValue":
        if not isinstance(other, AValue):
            other = type(self)(other)
        self._value = self._clip(self._value - other._value)
        return self

    def __imul__(self, other: Union["AValue", int, Tuple[int, ...], ndarray, None]) -> "AValue":
        if not isinstance(other, AValue):
            other = type(self)(other)
        self._value = self._clip(self._value * other._value)
        return self

    def __itruediv__(self, other: Union["AValue", int, Tuple[int, ...], ndarray, None]) -> "AValue":
        if not isinstance(other, AValue):
            other = type(self)(other)
        if other.is_zero:
            self._value = self.infvalue
        else:
            self._value = self._clip(self._value / other._value)
        return self


class ADD:
    def __init__(self, variables: List[int], diameter: int, atype: Type[AValue]) -> None:
        self.variables = variables
        self.diameter = diameter
        self.atype = atype
        self.root = 0
        self.nodes = np.zeros((len(variables), diameter))
        self.cleft = np.zeros((len(variables), diameter), dtype=int)
        self.cright = np.zeros((len(variables), diameter), dtype=int)
        self.aleft = np.full((len(variables), diameter), fill_value=atype(0), dtype=atype)
        self.aright = np.full((len(variables), diameter), fill_value=atype(0), dtype=atype)

    def __call__(self, *args: bool) -> AValue:
        if len(args) != len(self.variables):
            raise ValueError("Exactly %d arguments expected but %d provided." % (len(self.variables), len(args)))

        j = self.root
        result = self.atype.get_zero()
        for i, arg in enumerate(args):
            result += self.aright[i, j] if arg else self.aleft[i, j]
            j = self.cright[i, j] if arg else self.cleft[i, j]

        return result

    def restrict(self, variable: int, value: bool, inplace: bool = False) -> "ADD":
        result = self if inplace else copy.deepcopy(self)
        idx = self.variables.index(variable)
        self.variables.pop(idx)
        child = self.cright if value else self.cleft
        adder = self.aright if value else self.aleft
        if idx > 0:
            pidx = idx - 1
            for i in range(self.diameter):
                if result.nodes[pidx, i] == 1:
                    result.aleft[pidx, i] += adder[idx, self.cleft[pidx, i]]
                    result.aright[pidx, i] += adder[idx, self.cleft[pidx, i]]
                    result.cleft[pidx, i] = child[idx, self.cleft[pidx, i]]
                    result.cright[pidx, i] = child[idx, self.cleft[pidx, i]]
        else:
            result.root = child[idx, self.root]
        # TODO: Prune dead nodes.
        result.nodes = np.delete(result.nodes, idx, axis=0)
        result.cleft = np.delete(result.cleft, idx, axis=0)
        result.cright = np.delete(result.cright, idx, axis=0)
        result.aleft = np.delete(result.aleft, idx, axis=0)
        result.aright = np.delete(result.aright, idx, axis=0)
        return result

    def sum(self, other: "ADD") -> "ADD":
        assert self.variables == other.variables
        result = ADD(self.variables, self.diameter * other.diameter, self.atype)
        pnodes: Dict[Tuple[int, int], int] = {}
        cnodes: Dict[Tuple[int, int], int] = {(self.root, other.root): 0}
        for idx in range(len(self.variables)):
            pnodes = cnodes
            cnodes = {}
            for (i, j), k in pnodes.items():
                lidx = cnodes.setdefault((self.cleft[idx, i], other.cleft[idx, j]), len(cnodes))
                ridx = cnodes.setdefault((self.cright[idx, i], other.cright[idx, j]), len(cnodes))
                result.nodes[idx, k] = 1
                result.cleft[idx, k] = lidx
                result.cright[idx, k] = ridx
                result.aleft[idx, k] = self.aleft[idx, i] + other.aleft[idx, j]
                result.aright[idx, k] = self.aright[idx, i] + other.aright[idx, j]
        return result

    def modelcount(self) -> ndarray:
        adomain = np.array(list(self.atype.domain()), dtype=self.atype)
        result_current = np.zeros((self.diameter, adomain.shape[0]), dtype=int)
        result_current[:, 0] = 1
        for i in reversed(range(len(self.variables))):
            result_previous = result_current
            result_current = np.zeros((self.diameter, adomain.shape[0]), dtype=int)

            for j in range(self.diameter):
                for k, e in enumerate(adomain):
                    result_current[j, k] = (
                        result_previous[self.cleft[i, j], index(e - self.aleft[i, j])]
                        + result_previous[self.cright[i, j], index(e - self.aright[i, j])]
                    ) * self.nodes[i, j]

        result = result_current[self.root]
        result[-1] = 2 ** len(self.variables) - sum(result)  # Correct the count of invalid values.
        return result

    @classmethod
    def construct_tree(cls, variables: List[int], atype: Type[AValue]) -> "ADD":
        diameter = 2 ** (len(variables) - 1)
        d = ADD(variables=variables, diameter=diameter, atype=atype)
        for i in range(len(variables) - 1):
            d.nodes[i, : 2 ** i] = np.ones(2 ** i)
            d.cleft[i, : 2 ** i] = np.arange(0, 2 ** (i + 1), 2)
            d.cright[i, : 2 ** i] = np.arange(1, 2 ** (i + 1) + 1, 2)
        d.nodes[len(variables) - 1] = 1
        return d

    @classmethod
    def construct_chain(cls, variables: List[int], atype: Type[AValue]) -> "ADD":
        d = ADD(variables=variables, diameter=1, atype=atype)
        d.nodes = np.ones((len(variables), 1))
        return d

    @classmethod
    def concatenate(cls, elements: List["ADD"]) -> "ADD":
        # Prepare the resulting diagram.
        diameter = max(x.diameter for x in elements)
        variables: List[int] = sum([x.variables for x in elements], [])
        atype = elements[0].atype
        result = ADD(variables=variables, diameter=diameter, atype=atype)
        result.root = elements[0].root

        # Perform the actual concatenation.
        margins = [diameter - x.diameter for x in elements]
        np.concatenate([np.pad(x.nodes, [(0, 0), (0, margins[i])]) for i, x in enumerate(elements)], out=result.nodes)
        np.concatenate([np.pad(x.cleft, [(0, 0), (0, margins[i])]) for i, x in enumerate(elements)], out=result.cleft)
        np.concatenate([np.pad(x.cright, [(0, 0), (0, margins[i])]) for i, x in enumerate(elements)], out=result.cright)
        np.concatenate(
            [np.pad(x.aleft, [(0, 0), (0, margins[i])], constant_values=atype(0)) for i, x in enumerate(elements)],
            out=result.aleft,
        )
        np.concatenate(
            [np.pad(x.aright, [(0, 0), (0, margins[i])], constant_values=atype(0)) for i, x in enumerate(elements)],
            out=result.aright,
        )

        # Reroute child connections between elements.
        idx = -1
        for i in range(len(elements) - 1):
            idx += len(elements[i].variables)
            selector = result.nodes[idx].astype("bool")
            result.cleft[idx, selector] = elements[i + 1].root
            result.cright[idx, selector] = elements[i + 1].root

        return result

    @classmethod
    def stack(cls, factors: List[int], elements: Dict[Tuple[bool, ...], "ADD"]) -> "ADD":

        if len(elements) != 2 ** len(factors):
            raise ValueError(
                "Given %d factors, the number of elements has to be exactly %d."
                % (len(factors), 2 ** (len(factors) + 1))
            )

        # Prepare the resulting diagram.
        diameter = sum(x.diameter for x in elements.values())
        variables: List[int] = factors + next(iter(elements.values())).variables
        atype = next(iter(elements.values())).atype
        result = ADD(variables=variables, diameter=diameter, atype=atype)

        # Construct the elements list according to their corresponding factor values.
        elements_list: List["ADD"] = []
        for value in product(*[[False, True] for _ in range(len(factors))]):
            e = elements.get(value, None)
            if e is None:
                raise ValueError("Element for valuation %s not provided." % str(value))
            elements_list.append(e)

        # Construct the header tree made up of factors.
        for i in range(len(factors) - 1):
            result.nodes[i, : 2 ** i] = np.ones(2 ** i)
            result.cleft[i, : 2 ** i] = np.arange(0, 2 ** (i + 1), 2)
            result.cright[i, : 2 ** i] = np.arange(1, 2 ** (i + 1) + 1, 2)
        result.nodes[len(factors) - 1, : 2 ** (len(factors) - 1)] = 1

        # Route factor leaves to element roots.
        offsets = np.zeros(len(elements_list), dtype=int)
        np.cumsum([elements_list[i].diameter for i in range(len(elements_list) - 1)], out=offsets[1:])
        roots = np.array([elements_list[i].root for i in range(len(elements_list))]) + offsets
        nf = len(factors)
        ne = len(elements)
        result.cleft[nf - 1, : 2 ** (nf - 1)] = [roots[i] for i in range(ne) if i % 2 == 0]  # noqa: E203
        result.cright[nf - 1, : 2 ** (nf - 1)] = [roots[i] for i in range(ne) if i % 2 == 1]  # noqa: E203

        # Perform the actual stacking of elements. We need to apply offsets to child node pointers.
        np.concatenate([x.nodes for x in elements_list], axis=1, out=result.nodes[len(factors) :])  # noqa: E203
        np.concatenate(
            [x.cleft + offsets[i] for i, x in enumerate(elements_list)],
            axis=1,
            out=result.cleft[len(factors) :],  # noqa: E203
        )
        np.concatenate(
            [x.cright + offsets[i] for i, x in enumerate(elements_list)],
            axis=1,
            out=result.cright[len(factors) :],  # noqa: E203
        )
        np.concatenate([x.aleft for x in elements_list], axis=1, out=result.aleft[len(factors) :])  # noqa: E203
        np.concatenate([x.aright for x in elements_list], axis=1, out=result.aright[len(factors) :])  # noqa: E203

        return result

    def __repr__(self) -> str:
        return "ROOT: %d\n" % self.root + "\n".join(
            "[% 2d]: %s"
            % (
                v,
                " ".join(
                    "[%d](%d, %s)(%d, %s)"
                    % (j, self.cleft[i, j], str(self.aleft[i, j]), self.cright[i, j], str(self.aright[i, j]))
                    for j, n in enumerate(self.nodes[i])
                    if n != 0
                ),
            )
            for i, v in enumerate(self.variables)
        )
