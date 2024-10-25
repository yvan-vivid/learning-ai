from dataclasses import dataclass
from functools import cached_property
from itertools import product
from typing import Iterable, Iterator, List, Self, Tuple

from torch import Tensor, int32, multinomial, zeros

from ..bigrams import gen_bigrams
from ..encoding.character import Token
from ..util import sliding_window


def _sample_model(probs: Tensor) -> int:
    return int(multinomial(probs, num_samples=1, replacement=True).item())


@dataclass(frozen=True)
class FreqModel:
    under: Tensor
    boundary: Token
    count_reg: int = 0

    @cached_property
    def probs(self) -> Tensor:
        p = (self.under + self.count_reg).float()
        return p / p.sum(1, keepdim=True)

    def loss(self, tokens: Iterable[Token]) -> float:
        total, count = 0.0, 0
        for pair in sliding_window(tokens, 2):
            if (p := self.probs[pair]) == 0:
                raise ValueError(f"Impossible token pair '{pair}' sampled")
            total += p.log().item()
            count += 1
        return -total / count

    def generate(self, max_length: int = 100) -> List[Token]:
        current = self.boundary
        name = []
        probs = self.probs
        for k in range(max_length):
            if (ix := _sample_model(probs[current])) == self.boundary:
                break
            name.append(ix)
            current = ix
        return name

    def items(self) -> Iterator[Tuple[Token, Token, int]]:
        for i, j in product(range(self.under.shape[0]), range(self.under.shape[1])):
            yield i, j, int(self.under[i, j].item())

    @classmethod
    def from_words(cls, token_count: int, boundary: Token, words: Iterable[List[Token]]) -> Self:
        freq = zeros((token_count, token_count), dtype=int32)
        for word in words:
            for bg in gen_bigrams(word):
                freq[bg] += 1
        return cls(freq, boundary)
