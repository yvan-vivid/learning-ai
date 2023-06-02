from micrograd.value import ValueGraph
from micrograd.assignment import Assignment


def test_create_complete_assignment():
    G = ValueGraph()
    x, y, z = G('x'), G('y'), G('z')
    w = x + y | 'w'
    _ = w * z | 'v'

    a = Assignment.create(G, {x: 5, y: 6, z: 7})
    assert a.is_complete()


def test_create_complete_assignment_with_constants():
    G = ValueGraph()
    x, y = G('x'), G('y')
    w = x + y | 'w'
    _ = w * 5 | 'v'

    a = Assignment.create(G, {x: 5, y: 6})
    assert a.is_complete()


def test_create_incomplete_assignment():
    G = ValueGraph()
    x, y, z = G('x'), G('y'), G('z')
    w = x + y | 'w'
    _ = w * z | 'v'

    a = Assignment.create(G, {x: 5, z: 7})
    assert not a.is_complete()
