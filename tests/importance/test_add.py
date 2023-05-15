import numpy as np
import pytest

from datascope.utility import AValue, ADD
from itertools import product
from operator import index


@pytest.mark.parametrize(
    "value,expected",
    [(0, (0, 0, 0)), (5, (5, 5, 5)), ((1, 1, 1), (1, 1, 1)), (np.array([2, 2, 2]), (2, 2, 2)), (None, None)],
)
def test_avalue_init_valid_1(value, expected):
    avalue = AValue[9, 9, 9](value)
    assert avalue.value == expected


def test_avalue_init_valid_2():
    assert AValue[9, 9, 9](3, 3, 3).value == (3, 3, 3)


def test_avalue_init_invalid_1():
    assert AValue.maxvalue == ()

    with pytest.raises(ValueError):
        AValue((1, 1, 1))

    with pytest.raises(ValueError):
        AValue[2, 2, 2]((1, 1, 1), 1, 1)

    with pytest.raises(ValueError):
        AValue[2, 2, 2]((1, 1))


def test_avalue_factories_1():
    assert AValue[3, 3, 3].get_zero().value == (0, 0, 0)

    assert AValue[3, 3, 3].get_inf().value is None

    assert AValue[3, 3, 3].get_basis(2).value == (0, 0, 1)


def test_avalue_repr_1():
    assert repr(AValue[3, 3, 3](0)) == "AValue[3, 3, 3](0)"

    assert repr(AValue[3, 3, 3](None)) == "AValue[3, 3, 3](None)"

    assert repr(AValue[3, 3, 3](3, 3, 3)) == "AValue[3, 3, 3](3, 3, 3)"


def test_avalue_domain_1():
    AV = AValue[2, 2]
    expected = [
        AV(0, 0),
        AV(0, 1),
        AV(0, 2),
        AV(1, 0),
        AV(1, 1),
        AV(1, 2),
        AV(2, 0),
        AV(2, 1),
        AV(2, 2),
        AV(None),
    ]
    assert list(AV.domain()) == expected


def test_avalue_index_1():
    AV = AValue[2, 2]
    domain = AV.domain()
    for i, av in enumerate(domain):
        assert index(av) == i


def test_avalue_index_2():
    AV = AValue[5, 4, 3, 2]
    domain = AV.domain()
    for i, av in enumerate(domain):
        assert index(av) == i


def test_avalue_hash_1():
    assert hash(AValue[3, 3, 3](None)) == hash(None)


def test_avalue_bool_1():
    assert not AValue[3, 3, 3](None)
    assert AValue[3, 3, 3](0)


def test_avalue_eq_1():
    assert AValue[3, 3, 3](0) == 0
    assert AValue[3, 3, 3](None) != 0
    with pytest.raises(ValueError):
        AValue[3, 3, 3](0) == 1.0


ops_tuples = [
    ("+", (0, 0), 0, (0, 0)),
    ("+", 0, (1, 1), (1, 1)),
    ("+", (0, 0), (0, 1), (0, 1)),
    ("+", (1, 1), (2, 2), None),
    ("+", (0, 1), (0, 2), None),
    ("-", (0, 0), 0, (0, 0)),
    ("-", (2, 2), 1, (1, 1)),
    ("-", 1, (0, 1), (1, 0)),
    ("-", (1, 1), (2, 2), None),
    ("*", (0, 0), 0, (0, 0)),
    ("*", (1, 1), 1, (1, 1)),
    ("*", 1, (0, 1), (0, 1)),
    ("*", (2, 2), (2, 2), None),
    ("/", (0, 0), 1, (0, 0)),
    ("/", (1, 1), 1, (1, 1)),
    ("/", 2, (2, 2), (1, 1)),
    ("/", (2, 2), 0, None),
]


@pytest.mark.parametrize("op,arg1,arg2,res", ops_tuples)
def test_avalue_ops_1(op, arg1, arg2, res):
    V22 = AValue[2, 2]
    varg1, varg2, vres = (
        V22(arg1) if isinstance(arg1, tuple) else arg1,
        V22(arg2) if isinstance(arg2, tuple) else arg2,
        V22(res) if isinstance(res, tuple) else res,
    )

    if op == "+":
        assert varg1 + varg2 == vres
    elif op == "-":
        assert varg1 - varg2 == vres
    elif op == "*":
        assert varg1 * varg2 == vres
    elif op == "/":
        assert varg1 / varg2 == vres


@pytest.mark.parametrize("op,arg1,arg2,res", ops_tuples)
def test_avalue_iops_1(op, arg1, arg2, res):
    V22 = AValue[2, 2]
    varg1, varg2, vres = V22(arg1), V22(arg2) if isinstance(arg2, tuple) else arg2, V22(res)

    if op == "+":
        varg1 += varg2
    elif op == "-":
        varg1 -= varg2
    elif op == "*":
        varg1 *= varg2
    elif op == "/":
        varg1 /= varg2

    assert varg1 == vres


def test_add_chain_1():
    V22 = AValue[2, 2]
    variables = [0, 1, 2, 3]
    add = ADD.construct_chain(variables, atype=V22)

    add.adder[0, 0, 1] = V22(0, 1)
    add.adder[1, 0, 1] = V22(0, 1)
    add.adder[2, 0, 1] = V22(0, 1)
    add.adder[3, 0, 1] = V22(0, 1)

    assert add(False, False, False, False) == V22(0)
    assert add(False, True, False, False) == V22(0, 1)
    assert add(False, True, True, False) == V22(0, 2)
    assert add(False, True, True, True) == V22(None)

    expected = np.zeros(V22.domainsize, dtype=int)
    expected[index(V22(0, 0))] = 1
    expected[index(V22(0, 1))] = 4
    expected[index(V22(0, 2))] = 6
    expected[index(V22(None))] = 5
    modelcount = add.modelcount()
    assert modelcount.tolist() == expected.tolist()


def test_add_tree_1():
    V22 = AValue[2, 2]
    variables = [0, 1, 2, 3]
    add = ADD.construct_tree(variables, atype=V22)

    add.adder[0, 0, 1] = V22(0, 1)
    add.adder[1, 0, 1] = V22(0, 1)
    add.adder[2, 0, 1] = V22(0, 1)
    add.adder[3, 0, 1] = V22(0, 1)

    assert add(False, False, False, False) == V22(0)
    assert add(True, False, False, False) == V22(0, 1)
    assert add(True, True, False, False) == V22(0, 1)
    assert add(False, True, False, False) == V22(0, 1)
    assert add(False, False, True, True) == V22(0, 1)

    expected = np.zeros(V22.domainsize, dtype=int)
    expected[index(V22(0, 0))] = 1
    expected[index(V22(0, 1))] = 15
    modelcount = add.modelcount()
    assert modelcount.tolist() == expected.tolist()


def test_add_concatenate_1():
    V22 = AValue[2, 2]
    variables_1 = [0, 1]
    add_1 = ADD.construct_chain(variables_1, atype=V22)
    add_1.adder[0, 0, 1] = V22(0, 1)
    add_1.adder[1, 0, 1] = V22(0, 1)

    variables_2 = [2, 3]
    add_2 = ADD.construct_chain(variables_2, atype=V22)
    add_2.adder[0, 0, 1] = V22(0, 1)
    add_2.adder[1, 0, 1] = V22(0, 1)

    add_r1 = ADD.concatenate([add_1, add_2])

    variables_r2 = variables_1 + variables_2
    add_r2 = ADD.construct_chain(variables_r2, atype=V22)
    for i in range(len(variables_r2)):
        add_r2.adder[i, 0, 1] = V22(0, 1)

    for values in product(*[[False, True] for _ in range(len(variables_r2))]):
        assert add_r1(*values) == add_r2(*values)


def test_add_stack_1():
    V22 = AValue[2, 2]
    variables = [1, 2]
    add_1 = ADD.construct_chain(variables, atype=V22)
    add_1.adder[0, 0, 0] = V22(0, 1)
    add_1.adder[1, 0, 0] = V22(0, 1)
    add_2 = ADD.construct_chain(variables, atype=V22)
    add_2.adder[0, 0, 1] = V22(0, 1)
    add_2.adder[1, 0, 1] = V22(0, 1)

    add = ADD.stack([0], {(0,): add_1, (1,): add_2})
    assert add(0, 0, 0) == V22(0, 2)
    assert add(1, 0, 0) == V22(0, 0)
    assert add(1, 1, 0) == V22(0, 1)
    assert add(0, 1, 0) == V22(0, 1)
    assert add(0, 0, 1) == V22(0, 1)
    assert add(0, 1, 1) == V22(0, 0)

    expected = np.zeros(V22.domainsize, dtype=int)
    expected[index(V22(0, 0))] = 2
    expected[index(V22(0, 1))] = 4
    expected[index(V22(0, 2))] = 2
    expected[index(V22(None))] = 0
    modelcount = add.modelcount()
    assert modelcount.tolist() == expected.tolist()


def test_add_stack_2():
    V22 = AValue[2, 2]
    variables = [2, 3]
    add_1 = ADD.construct_chain(variables, atype=V22)
    add_1.adder[0, 0, 0] = V22(0, 1)
    add_1.adder[1, 0, 0] = V22(0, 1)
    add_2 = ADD.construct_chain(variables, atype=V22)
    add_2.adder[0, 0, 1] = V22(0, 1)
    add_2.adder[1, 0, 1] = V22(0, 1)
    add_3 = ADD.construct_chain(variables, atype=V22)

    add = ADD.stack([0, 1], {(False, False): add_1, (True, True): add_2, (True, False): add_3, (False, True): add_3})
    assert add(False, False, False, False) == V22(0, 2)
    assert add(True, True, False, False) == V22(0, 0)
    assert add(True, True, True, False) == V22(0, 1)
    assert add(False, False, True, False) == V22(0, 1)
    assert add(False, False, False, True) == V22(0, 1)
    assert add(False, False, True, True) == V22(0, 0)


def test_add_stack_invalid_1():
    V22 = AValue[2, 2]
    variables = [1, 2]
    add_1 = ADD.construct_chain(variables, atype=V22)
    add_1.adder[0, 0, 0] = V22(0, 1)
    add_1.adder[1, 0, 0] = V22(0, 1)
    add_2 = ADD.construct_chain(variables, atype=V22)
    add_2.adder[0, 0, 1] = V22(0, 1)
    add_2.adder[1, 0, 1] = V22(0, 1)

    with pytest.raises(ValueError):
        ADD.stack([0, 1], {(False,): add_1, (True,): add_2})

    with pytest.raises(ValueError):
        ADD.stack([0], {(False,): add_1, (True, True): add_2})


def test_add_call_invalid_1():
    V22 = AValue[2, 2]
    variables = [0, 1, 2, 3]
    add = ADD.construct_chain(variables, atype=V22)

    with pytest.raises(ValueError):
        add(False, False, False)


def test_add_repr_1():
    V22 = AValue[2, 2]
    variables = [0, 1, 2, 3]
    add = ADD.construct_chain(variables, atype=V22)
    lines = str.split(repr(add), sep="\n")
    assert len(lines) == len(variables) + 1


def test_add_restrict_1():
    V22 = AValue[2, 2]
    variables = [2, 3]
    add_1 = ADD.construct_chain(variables, atype=V22)
    add_1.adder[0, 0, 0] = V22(0, 1)
    add_1.adder[1, 0, 0] = V22(0, 1)
    add_2 = ADD.construct_chain(variables, atype=V22)
    add_2.adder[0, 0, 1] = V22(0, 1)
    add_2.adder[1, 0, 1] = V22(0, 1)
    add_3 = ADD.construct_chain(variables, atype=V22)

    add = ADD.stack([0, 1], {(False, True): add_1, (True, True): add_2, (True, False): add_3, (False, False): add_3})

    add.restrict(1, True, inplace=True)

    assert add(False, False, False) == V22(0, 2)
    assert add(True, False, False) == V22(0, 0)
    assert add(True, True, False) == V22(0, 1)
    assert add(False, True, False) == V22(0, 1)
    assert add(False, False, True) == V22(0, 1)
    assert add(False, True, True) == V22(0, 0)


def test_add_restrict_2():
    V22 = AValue[2, 2]
    variables = [2, 3]
    add_1 = ADD.construct_chain(variables, atype=V22)
    add_1.adder[0, 0, 0] = V22(0, 1)
    add_1.adder[1, 0, 0] = V22(0, 1)
    add_2 = ADD.construct_chain(variables, atype=V22)
    add_2.adder[0, 0, 1] = V22(0, 1)
    add_2.adder[1, 0, 1] = V22(0, 1)
    add_3 = ADD.construct_chain(variables, atype=V22)

    add = ADD.stack([0, 1], {(True, False): add_1, (True, True): add_2, (False, True): add_3, (False, False): add_3})

    add.restrict(0, True, inplace=True)

    assert add(False, False, False) == V22(0, 2)
    assert add(True, False, False) == V22(0, 0)
    assert add(True, True, False) == V22(0, 1)
    assert add(False, True, False) == V22(0, 1)
    assert add(False, False, True) == V22(0, 1)
    assert add(False, True, True) == V22(0, 0)


def test_add_sum_1():
    V22 = AValue[2, 2]
    variables = [0, 1]
    add_1 = ADD.construct_chain(variables, atype=V22)
    add_1.adder[0, 0, 1] = V22(0, 1)
    add_2 = ADD.construct_chain(variables, atype=V22)
    add_2.adder[1, 0, 1] = V22(0, 1)

    add = add_1.sum(add_2)

    assert add(False, False) == V22(0, 0)
    assert add(False, True) == V22(0, 1)
    assert add(True, False) == V22(0, 1)
    assert add(True, True) == V22(0, 2)


def test_add_sum_2():
    V22 = AValue[2, 2]
    add_11_a = ADD.construct_chain([1, 2], atype=V22)
    add_11_a.adder[0, 0, 1] = V22(0, 1)
    add_11_a.adder[1, 0, 1] = V22(0, 1)
    add_12_a = ADD.construct_chain([1, 2], atype=V22)
    add_1_a = ADD.stack([0], {(True,): add_11_a, (False,): add_12_a})

    add_21_a = ADD.construct_chain([4], atype=V22)
    add_21_a.adder[0, 0, 1] = V22(0, 1)
    add_22_a = ADD.construct_chain([4], atype=V22)
    add_2_a = ADD.stack([3], {(True,): add_21_a, (False,): add_22_a})

    add_a = ADD.concatenate([add_1_a, add_2_a])

    add_1_b = ADD.construct_chain([0, 1], atype=V22)
    add_21_b = ADD.construct_chain([3, 4], atype=V22)
    add_22_b = ADD.construct_chain([3, 4], atype=V22)
    add_22_b.adder[0, 0, 0] = V22(None)
    add_22_b.adder[1, 0, 0] = V22(None)
    add_2_b = ADD.stack([2], {(True,): add_21_b, (False,): add_22_b})

    add_b = ADD.concatenate([add_1_b, add_2_b])

    add = add_a.sum(add_b)

    assert add(False, False, False, False, False) == V22(None)
    assert add(True, True, False, True, False) == V22(None)
    assert add(True, True, False, True, False) == V22(None)
    assert add(True, True, False, True, True) == V22(0, 2)
    assert add(True, True, True, True, True) == V22(None)
