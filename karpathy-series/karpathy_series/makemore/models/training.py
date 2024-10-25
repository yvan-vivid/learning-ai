from typing import Iterable, List, Tuple, TypeAlias

from torch import Tensor, tensor

from ..bigrams import gen_bigrams
from ..encoding.character import Token

TrainingSequence: TypeAlias = Iterable[Tuple[Tensor, Tensor]]


def make_training(words: Iterable[List[Token]]) -> Tuple[Tensor, Tensor]:
    xs, ys = [], []
    for word in words:
        for x, y in gen_bigrams(word):
            xs.append(x)
            ys.append(y)
    return (tensor(xs), tensor(ys))
