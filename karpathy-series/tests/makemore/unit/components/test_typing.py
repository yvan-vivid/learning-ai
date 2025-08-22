from karpathy_series.makemore.components.typing import Dims, Shape


def test_replace_active() -> None:
    assert Shape.from_ints([]).replace_active(Dims(5)) == Shape.from_ints([])
    assert Shape.from_ints([3]).replace_active(Dims(5)) == Shape.from_ints([5])
    assert Shape.from_ints([1, 2, 3]).replace_active(Dims(5)) == Shape.from_ints([1, 2, 5])


def test_expand() -> None:
    assert Shape.from_ints([]).expand(Dims(5)) == Shape.from_ints([5])
    assert Shape.from_ints([1, 2, 3]).expand(Dims(5)) == Shape.from_ints([1, 2, 3, 5])


def test_extend() -> None:
    assert Shape.from_ints([]).extend() == Shape.from_ints([1])
    assert Shape.from_ints([]).extend(0) == Shape.from_ints([1])
    assert Shape.from_ints([5]).extend(0) == Shape.from_ints([1, 5])
    assert Shape.from_ints([5]).extend(1) == Shape.from_ints([5, 1])
    assert Shape.from_ints([5]).extend(-1) == Shape.from_ints([1, 5])
    assert Shape.from_ints([5]).extend() == Shape.from_ints([5, 1])


def test_slide() -> None:
    assert Shape.from_ints([6]).slide(0, Dims(3)) == Shape.from_ints([2, 3])
    assert Shape.from_ints([6, 12]).slide(0, Dims(3)) == Shape.from_ints([2, 36])
    assert Shape.from_ints([6, 12]).slide(-2, Dims(3)) == Shape.from_ints([2, 36])
    assert Shape.from_ints([6, 12]).slide(1, Dims(3)) == Shape.from_ints([6, 4, 3])
    assert Shape.from_ints([6, 12]).slide(-1, Dims(3)) == Shape.from_ints([6, 4, 3])
