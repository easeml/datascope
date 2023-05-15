import numpy as np

from copy import deepcopy
from functools import partial
from itertools import combinations, chain, product
from math import comb
from numpy.typing import NDArray
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from typing import Type, Hashable, List, Dict, Set, Tuple, Optional, Union, Iterable, Sequence

from ..utility import AValue, ADD, Provenance


class ATally(AValue):
    numtuples: int
    numneighbors: int
    numclasses: int
    slots_with: NDArray
    slots_without: NDArray
    domain_index: Optional[Dict[Optional[Tuple[int, ...]], int]] = None

    def __init__(
        self,
        tupletally: Union[int, Tuple[int, ...], NDArray, None],
        labeltally_with: Tuple[int, ...] = (),
        labeltally_without: Tuple[int, ...] = (),
    ) -> None:
        if isinstance(tupletally, int) and len(labeltally_with) > 0 and len(labeltally_without) > 0:
            super().__init__((tupletally,) + labeltally_with + labeltally_without)
        else:
            super().__init__(tupletally)

    def _clip(self, value: NDArray) -> NDArray:
        if (
            np.any(value < 0)
            or value[0] > self.numtuples
            or np.sum(value[self.slots_with]) > self.numneighbors
            or np.sum(value[self.slots_without]) > self.numneighbors
        ):
            return self.infvalue
        return value

    def __class_getitem__(cls, key: Tuple[int, ...]) -> Type["ATally"]:
        numtuples, numneighbors, numclasses = key

        maxvalue = tuple([numtuples] + [numneighbors] * numclasses * 2)
        slots_with = np.arange(1, numclasses + 1, dtype=int)
        slots_without = np.arange(numclasses + 1, numclasses * 2 + 1, dtype=int)
        infvalue = np.array(maxvalue) + 1
        zerovalue = np.zeros_like(infvalue)
        domainsize = (numtuples + 1) * comb(numneighbors + numclasses, numclasses) ** 2 + 1
        result = type(
            cls.__name__,
            (cls,),
            dict(
                maxvalue=maxvalue,
                infvalue=infvalue,
                zerovalue=zerovalue,
                domainsize=domainsize,
                numtuples=numtuples,
                numneighbors=numneighbors,
                numclasses=numclasses,
                slots_with=slots_with,
                slots_without=slots_without,
            ),
        )
        return result

    @classmethod
    def domain_single(cls) -> Iterable[Tuple[int, ...]]:
        for x in product(*[range(cls.numneighbors + 1) for _ in range(cls.numclasses)]):
            if sum(x) <= cls.numneighbors:
                yield x

    @classmethod
    def domain(cls) -> Iterable["ATally"]:
        for tupletally, labeltally_with, labeltally_without in product(
            range(cls.numtuples + 1), cls.domain_single(), cls.domain_single()
        ):
            yield cls(tupletally, labeltally_with, labeltally_without)
        yield cls(None)

    def __index__(self) -> int:
        cls = type(self)
        if cls.domain_index is None:
            cls.domain_index = dict((v.value, i) for i, v in enumerate(cls.domain()))
        return cls.domain_index[self.value]

    def __repr__(self) -> str:
        return "%s[%d, %d, %d]%s" % (
            self.__class__.__name__,
            self.numtuples,
            self.numneighbors,
            self.numclasses,
            str(self),
        )

    def __str__(self) -> str:
        if self.is_inf:
            return "(None)"
        elif self.is_zero:
            return "(0)"
        else:
            return "(%s, %s, %s)" % (str(self.tupletally), str(self.labeltally_with), str(self.labeltally_without))

    @property
    def tupletally(self) -> Optional[int]:
        if np.array_equal(self._value, self.infvalue):
            return None
        else:
            return self._value[0]

    @property
    def labeltally_with(self) -> Optional[Tuple[int, ...]]:
        if np.array_equal(self._value, self.infvalue):
            return None
        else:
            return tuple(self._value[self.slots_with])

    @property
    def labeltally_without(self) -> Optional[Tuple[int, ...]]:
        if np.array_equal(self._value, self.infvalue):
            return None
        else:
            return tuple(self._value[self.slots_without])


def compile(provenance: Provenance, atype: Type[ATally]) -> Tuple[ADD, List[List[Tuple]]]:
    if provenance.max_disjunctions > 1:
        raise ValueError("Provenance with disjunctions cannot be compiled into an ADD.")

    # Check if each polynomial has only one element.
    if provenance.max_conjunctions == 1:
        locations: List[List[Tuple]] = list(map(lambda x: [(x[0], 0, x[1])], map(tuple, provenance.data[:, 0, 0, 0:2])))
        add = ADD.construct_chain(
            units=list(range(provenance.num_units)), num_candidates=provenance.num_candidates, atype=atype
        )

        return add, locations

    else:
        # Extract all pairings of units that appear together in any expression.
        tuple_units = [np.sort(np.delete(a, np.asarray(a == -1).nonzero())) for a in provenance.data[:, 0, :, 0]]
        tuple_unit_pairs = map(partial(combinations, r=2), tuple_units)
        pairings = np.array(list(set(chain.from_iterable(tuple_unit_pairs))))

        # Compute the degree of every unit which represents the number of distinct units it appears together with.
        unique, unique_counts = np.unique(pairings, return_counts=True)
        degrees = np.zeros((provenance.num_units,), dtype=int)
        degrees[unique] = unique_counts

        # Compute the neighborhood of each unit which tracks all unit pairs that appear together.
        neighbors = csr_matrix(
            (np.repeat(1, repeats=pairings.shape[0]), (pairings[:, 0], pairings[:, 1])),
            shape=[provenance.num_units, provenance.num_units],
        )
        neighbors += neighbors.transpose()

        # Find components of the unit graph where each component will comprise of units that appear together.
        num_components, components_index = connected_components(neighbors, directed=False, return_labels=True)
        components: List[Set[int]] = [set() for _ in range(num_components)]
        for unit, component in enumerate(components_index):
            components[component].add(unit)

        # Find all leaf units that can be separted from factors. This is the maximal independent set problem.
        leaf_units = set()
        available_units = set(range(provenance.num_units))
        for unit in np.argsort(degrees):
            if unit in available_units:
                leaf_units.add(unit)
                available_units.difference_update(neighbors.getrow(unit).indices)

        # Construct the ADD from the provenance information.
        vertical_elements = []
        for component in components:
            factors = sorted(component - leaf_units)
            leaves = sorted(component & leaf_units)
            element = ADD.construct_chain(units=leaves, num_candidates=provenance.num_candidates, atype=atype)
            horizontal_elements = dict(
                (a, deepcopy(element))
                for a in product(*[range(provenance.num_candidates) for _ in range(len(factors))])
            )
            vertical_elements.append(ADD.stack(factors=factors, elements=horizontal_elements))
        add = ADD.concatenate(elements=vertical_elements)

        # For each tuple, compute the locations where its avalue lies in the ADD.
        locations = []
        for i in range(len(provenance)):
            assignments = filter(lambda x: x[0] != -1 and x[1] != -1, map(tuple, provenance.data[i, 0, :, :]))
            units, values = zip(*assignments)
            locations.append(add.get_update_location(units=units, values=values))

        return add, locations


class ShapleyOracle:
    def __init__(
        self,
        provenance: Provenance,
        labels: Union[Sequence[int], NDArray],
        distances: Union[Sequence[int], NDArray],
        atype: Type[ATally],
    ) -> None:
        # Compile the provenance into an ADD and obtain the node index which maps tuples to nodes.
        self._provenance = provenance
        self._atype = atype
        self._add, locations = compile(provenance=provenance, atype=atype)

        # For each tuple, assume its a boundary tuple and produce an ADD for tallying tuples sorted above it.
        self._add_with: Dict[Optional[int], ADD] = {}
        self._add_without: Dict[Optional[int], ADD] = {}
        for t in chain(range(len(provenance)), [None]):
            boundary_add_with = deepcopy(self._add)
            boundary_add_without = deepcopy(self._add)

            # Use locations to update the adder according to this tuple being the boundary tuple.
            for tt in range(len(provenance)):
                if t is None or distances[t] >= distances[tt]:
                    tally = tuple(np.eye(atype.numclasses, dtype=int)[labels[tt]])
                    zeros = (0,) * atype.numclasses
                    boundary_add_with.update(location=locations[tt], avalue=atype(0, tally, zeros), increment=True)
                    boundary_add_without.update(location=locations[tt], avalue=atype(0, zeros, tally), increment=True)

            # Make it invalid to exclude the tuple t from the dataset.
            if t is not None:
                boundary_units = list(filter(lambda x: x != -1, provenance.data[t, 0, :, 0]))
                for unit in boundary_units:
                    unit_location = self._add.get_update_location(units=[unit], values=[0])
                    boundary_add_with.update(location=unit_location, avalue=atype(None))
                    boundary_add_without.update(location=unit_location, avalue=atype(None))

            # Add the two ADD's to the cache.
            self._add_with[t] = boundary_add_with
            self._add_without[t] = boundary_add_without

    def query(
        self, target: Hashable, boundary_with: Optional[int], boundary_without: Optional[int]
    ) -> Dict[ATally, NDArray]:
        unit = self._provenance.units_index[target]

        add_with = self._add_with[boundary_with].restrict(unit, 1)
        add_without = self._add_without[boundary_without].restrict(unit, 0)
        add = add_with.sum(add_without)
        # add = (
        #     add_with.sum(add_without)
        #     if add_with is not None and add_without is not None
        #     else add_with
        #     if add_with is not None
        #     else add_without
        #     if add_without is not None
        #     else deepcopy(self._add)
        # )
        # add = self._add_with[boundary_with].restrict(unit,1).sum(self._add_without[boundary_without].restrict(unit,0))
        # locations = add.get_update_location(add.units, repeat(1))
        zeros = (0,) * self._atype.numclasses
        add.adder[:, :, 1] += self._atype(1, zeros, zeros)
        # add.update(location=locations, avalue=self._atype(1, zeros, zeros), increment=True)
        modelcounts: List[NDArray] = list(add.modelcount())
        result = dict((avalue, counts) for (avalue, counts) in zip(self._atype.domain(), modelcounts))
        return result
