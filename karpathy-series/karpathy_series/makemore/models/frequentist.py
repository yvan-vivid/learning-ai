from dataclasses import dataclass
from functools import cached_property
from itertools import product
from typing import Iterable, Iterator, Self, Tuple

from torch import Tensor, int32, zeros

from ..encoding.abstract import Encoder
from ..encoding.character import Token
from ..util import sample_index_model, sliding_window


@dataclass(frozen=True)
class FreqModel:
    under: Tensor
    in_encoder: Encoder[Token, str]
    out_encoder: Encoder[Token, str]
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

    def generate(self, boundary: str = ".", max_length: int = 100) -> str:
        start_code = self.in_encoder.encode(boundary)
        end_code = self.out_encoder.encode(boundary)
        current = start_code
        name = ""
        for k in range(max_length):
            if (current := sample_index_model(self.probs[current])) == end_code:
                break
            if (out := self.out_encoder.decode(current)) is None:
                continue
            name += out
        return name

    def items(self) -> Iterator[Tuple[Token, Token, int]]:
        for i, j in product(range(self.under.shape[0]), range(self.under.shape[1])):
            yield i, j, int(self.under[i, j].item())

    @classmethod
    def as_cleared(cls, in_encoder: Encoder[Token, str], out_encoder: Encoder[Token, str]) -> Self:
        return cls(zeros(in_encoder.size, out_encoder.size, dtype=int32), in_encoder, out_encoder)
