from karpathy_series.makemore.util import block_sequence, sliding_window, traverse_list, traverse_str


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


def test_block_sequence() -> None:
    assert tuple(block_sequence(3, 1)) == (slice(0, 1), slice(1, 2), slice(2, 3))
    assert tuple(block_sequence(3, 2)) == (slice(0, 2), slice(2, None))
    assert tuple(block_sequence(3, 3)) == (slice(0, 3),)
    assert tuple(block_sequence(3, 4)) == (slice(0, 3),)
    assert tuple(block_sequence(0, 4)) == ()
