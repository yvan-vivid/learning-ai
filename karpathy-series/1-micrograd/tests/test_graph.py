from micrograd.graph import Dag


def _test_graph() -> Dag:
    dag: Dag[None] = Dag()
    a = dag.node(None, label="a")
    b = dag.node(None, label="b")
    c = dag.node(None, a, b, label="c")
    _ = dag.node(None, a, c, label="d")
    _ = dag.node(None, a, label="e")
    f = dag.node(None, label="f")
    _ = dag.node(None, b, f, c, label="g")
    return dag


def test_entries():
    graph = _test_graph()
    assert set(r.label() for r in graph.entries()) == {"a", "b", "f"}


def test_roots():
    graph = _test_graph()
    assert set(r.label() for r in graph.roots()) == {"g", "e", "d"}


def test_topological():
    graph = _test_graph()
    assert list(r.label() for r in graph.topological()) == [
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "g",
    ]


def test_topological_reverse():
    graph = _test_graph()
    assert list(r.label() for r in graph.topological(reverse=True)) == [
        "g",
        "f",
        "e",
        "d",
        "c",
        "b",
        "a",
    ]
