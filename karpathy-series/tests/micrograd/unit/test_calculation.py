from karpathy_series.micrograd.assignment import Assignment
from karpathy_series.micrograd.calculation import GraphValuation
from karpathy_series.micrograd.valuation import Valuation
from karpathy_series.micrograd.value import ValueGraph


def test_initialize_valuation() -> None:
    G = ValueGraph()
    x, y, z = G["x", "y", "z"]
    w = x + y | "w"
    v = w * z | "v"

    a = Assignment.create(G, {x: 5, y: 6, z: 7})
    assert a.is_complete()

    gv = GraphValuation.initialize(a)
    assert gv.assigned == {
        x.node.ident: Valuation(5),
        y.node.ident: Valuation(6),
        z.node.ident: Valuation(7),
        w.node.ident: Valuation(),
        v.node.ident: Valuation(),
    }


def test_forward() -> None:
    G = ValueGraph()
    x, y, z = G["x", "y", "z"]
    w = x + y | "w"
    v = w * z | "v"
    h = v + x | "h"

    a = Assignment.create(G, {x: 5, y: 6, z: 7})
    assert a.is_complete()

    gv = GraphValuation.initialize(a)
    gv.forward()

    assert gv.assigned == {
        x.node.ident: Valuation(5),
        y.node.ident: Valuation(6),
        z.node.ident: Valuation(7),
        w.node.ident: Valuation(11),
        v.node.ident: Valuation(77),
        h.node.ident: Valuation(82),
    }


def test_backward() -> None:
    G = ValueGraph()
    x, y, z = G["x", "y", "z"]
    w = x + y | "w"
    v = w * z | "v"
    h = v + x | "h"

    a = Assignment.create(G, {x: 5, y: 6, z: 7})
    assert a.is_complete()

    gv = GraphValuation.initialize(a)
    gv.forward()
    gv.backward()

    assert gv.assigned == {
        x.node.ident: Valuation(5, 8),
        y.node.ident: Valuation(6, 7),
        z.node.ident: Valuation(7, 11),
        w.node.ident: Valuation(11, 7),
        v.node.ident: Valuation(77, 1),
        h.node.ident: Valuation(82, 1),
    }


def test_backward_converge() -> None:
    G = ValueGraph()
    x, y = G["x", "y"]
    w = x + y | "w"
    u = 2 * x | "u"
    v = w * u | "v"
    z = v + u | "z"
    h = z * u | "h"

    a = Assignment.create(G, {x: 5, y: 6})
    assert a.is_complete()

    gv = GraphValuation.initialize(a)
    gv.forward()
    gv.backward()

    assert gv.assigned == {
        x.node.ident: Valuation(5, 580),
        y.node.ident: Valuation(6, 100),
        w.node.ident: Valuation(11, 100),
        u.node.ident: Valuation(10, 240),
        v.node.ident: Valuation(110, 10),
        z.node.ident: Valuation(120, 10),
        h.node.ident: Valuation(1200, 1),
    }
