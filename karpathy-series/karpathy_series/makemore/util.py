from itertools import islice, tee
from typing import Iterable, Iterator, List, Optional, Tuple, TypeVar

from torch import Tensor, multinomial

V = TypeVar("V")


def sliding_window(itb: Iterable[V], n: int) -> Iterator[Tuple[V, ...]]:
    its = tee(itb, n)
    for i, it in enumerate(its):
        next(islice(it, i, i), None)
    return zip(*its)


def traverse_list(xs: Iterable[Optional[V]]) -> Optional[List[V]]:
    out = []
    for x in xs:
        if x is None:
            return None
        out.append(x)
    return out


def traverse_str(xs: Iterable[Optional[str]]) -> Optional[str]:
    out = ""
    for x in xs:
        if x is None:
            return None
        out += x
    return out


def softmax(v: Tensor) -> Tensor:
    p = v.exp()
    return p / p.sum(1, keepdim=True)


def sample_index_model(probs: Tensor) -> int:
    return int(multinomial(probs, num_samples=1, replacement=True).item())
