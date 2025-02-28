from itertools import islice, tee
from typing import Iterable, Iterator, Optional

from torch import Tensor, multinomial


def sliding_window[V](itb: Iterable[V], n: int) -> Iterator[tuple[V, ...]]:
    its = tee(itb, n)
    for i, it in enumerate(its):
        _ = next(islice(it, i, i), None)
    return zip(*its)


def traverse_list[V](xs: Iterable[Optional[V]]) -> Optional[list[V]]:
    out: list[V] = []
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


# Array utilities


def norm_distro(v: Tensor, dim: int = 0) -> Tensor:
    return v / v.sum(dim, keepdim=True)


def sample_index_model(probs: Tensor) -> int:
    return int(multinomial(probs, num_samples=1, replacement=True).item())


def sample_index_logits(logits: Tensor) -> int:
    return sample_index_model(logits.softmax(-1))


def cross_entropy_exp(u: Tensor, y: Tensor) -> Tensor:
    return -norm_distro(u, -1)[range(u.shape[0]), y].log().mean()
