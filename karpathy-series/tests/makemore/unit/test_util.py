from karpathy_series.makemore.util import sliding_window, traverse_list, traverse_str


def test_sliding_window() -> None:
    it = sliding_window(range(5), 3)
    assert next(it) == (0, 1, 2)
    assert next(it) == (1, 2, 3)
    assert next(it) == (2, 3, 4)


def test_traverse_list() -> None:
    assert traverse_list(()) == []
    assert traverse_list((1, None)) is None
    assert traverse_list((None, 1, 2)) is None
    assert traverse_list((1, None, 2)) is None
    assert traverse_list((1, 2, 3)) == [1, 2, 3]


def test_traverse_str() -> None:
    assert traverse_str(()) == ""
    assert traverse_str(("a", None)) is None
    assert traverse_str((None, "a", "b")) is None
    assert traverse_str(("a", None, "b")) is None
    assert traverse_str(("a", "b", "c")) == "abc"
