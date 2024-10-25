from itertools import starmap
from typing import Iterable, Iterator, NamedTuple

from .util import sliding_window


class BiGram(NamedTuple):
    a: int
    b: int


def gen_bigrams(tokens: Iterable[int]) -> Iterator[BiGram]:
    return starmap(BiGram, sliding_window(tokens, 2))
