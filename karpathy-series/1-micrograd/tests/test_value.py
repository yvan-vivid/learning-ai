from micrograd.value_type import Variable, Sum, Prod
from micrograd.value import ValueGraph


def test_construct_inputs():
    graph = ValueGraph()
    x = graph("x")
    y = graph("y")

    assert x.value_type == Variable()
    assert x.label() == "x"
    assert y.value_type == Variable()
    assert y.label() == "y"


def test_construct_binops():
    graph = ValueGraph()
    x = graph("x")
    y = graph("y")
    u = x + y | "u"
    v = x * y | "v"

    assert u.value_type == Sum()
    assert u.label() == "u"

    assert v.value_type == Prod()
    assert v.label() == "v"


def test_construct_binop_with_const():
    graph = ValueGraph()
    x = graph("x")
    u = x * 5 | "u"
    v = 5 * x | "v"

    assert u.value_type == Prod(5)
    assert u.label() == "u"

    assert v.value_type == Prod(5)
    assert v.label() == "v"
