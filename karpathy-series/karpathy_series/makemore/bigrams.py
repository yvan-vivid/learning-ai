from itertools import starmap
from typing import Iterator, NamedTuple, Self, Tuple

from .util import sliding_window


def _padding(pad: str, size: int, word: str) -> str:
    return (pad * size) + word + pad


def _padded_window(pad: str, size: int, word: str) -> Iterator[Tuple[str, ...]]:
    return sliding_window(_padding(pad, size, word), size + 1)


class BiGram(NamedTuple):
    a: str
    b: str

    @classmethod
    def generate(cls, pad: str, word: str) -> Iterator[Self]:
        return starmap(cls, _padded_window(pad, 1, word))


class TriGram(NamedTuple):
    a: Tuple[str, str]
    b: str

    @classmethod
    def generate(cls, pad: str, word: str) -> Iterator[Self]:
        for a, b, c in _padded_window(pad, 2, word):
            yield cls((a, b), c)
