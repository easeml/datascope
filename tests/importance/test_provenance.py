import numpy as np
import pytest

from datascope.utility.provenance import Units, Provenance


def test_units_init_and_properties_1():
    x = Units(name="x")
    assert len(x) == 0
    assert len(x.units) == 0
    assert len(x.candidates) == 0
    assert x.name == "x"
    assert repr(x["a"]) == "x['a']"
    assert len(x) == 1
    x = Units(units=5, candidates=5)
    assert len(x.units) == 5
    assert len(x.candidates) == 5
    assert 0 in x
    assert x.units_index[1] == 1
    assert x.candidates_index[1] == 1
    assert next(iter(x)) == 0
    assert x[0].key == 0
    assert x[0].parent is x
    x = Units(units=list(range(5)), candidates=list(range(5)))
    assert len(x.units) == 5
    assert len(x.candidates) == 5
    with pytest.raises(ValueError):
        x = Units(units=set(range(5)), candidates=5)
    with pytest.raises(ValueError):
        x = Units(units=5, candidates=set(range(5)))
    with pytest.raises(KeyError):
        x = Units(units=5)
        x[5]


def test_units_prefix_1():
    x = Units(units=[1, 2])
    x.prefix(prefix=0, inplace=True)
    assert (0, 1) in x and (0, 2) in x


def test_units_union_1():
    x = Units(units=[1, 2], candidates=[True, False])
    y = Units(units=[1, 2], candidates=[True, False, "Maybe"])
    z = x.union(other=y, prefix="x", other_prefix="y")
    assert ("x", 1) in z and ("y", 1) in z
    assert len(z.candidates) == 3


def test_provenance_query_1():
    provenance = Provenance(units=5, candidates=2)
    result = provenance.query({1: 1, 2: 1})
    assert np.sum(result) == 2


def test_provenance_query_2():
    x = Units(candidates=2)
    expressions = [x[0] == 0, x[1] == 1, x[2] == 1, x[3] == 0, x[4] == 1]
    provenance = Provenance(expressions=expressions)
    values = {1: 1, 2: 1}
    assert list(provenance.query(values=values)) == [e.eval(values=values) for e in expressions]


def test_provenance_query_3():
    x = Units(candidates=2)
    expressions = [(x[0] == 0) & (x[3] == 1), (x[1] == 1) & (x[3] == 0), (x[2] == 1) & (x[3] == 1)]
    provenance = Provenance(expressions=expressions)
    values = {1: 1, 2: 1}
    assert list(provenance.query(values=values, dtype=int)) == [
        i for i, e in enumerate(expressions) if e.eval(values=values)
    ]
