from datascope.importance.oracle import ATally
from operator import index


def test_atally_index_1():
    AV = ATally[5, 3, 2]
    domain = AV.domain()
    for i, av in enumerate(domain):
        assert index(av) == i
