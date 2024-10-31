from dataclasses import dataclass
from functools import cached_property
from itertools import product
from typing import Iterable, Iterator, Self, Tuple

from torch import Tensor, int32, zeros

from ..encoding.abstract import Encoder
from ..encoding.character import Token
from ..util import sample_index_model, sliding_window
from .generation import bi_gram_generate


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

    def generate(self, max_length: int = 100) -> str:
        def _forward(in_v: str) -> str:
            return self.out_encoder.decode_or_raise(
                sample_index_model(self.probs[self.in_encoder.encode_or_raise(in_v)])
            )

        return bi_gram_generate(_forward, ".", ".", max_length)

    def items(self) -> Iterator[Tuple[Token, Token, int]]:
        for i, j in product(range(self.under.shape[0]), range(self.under.shape[1])):
            yield i, j, int(self.under[i, j].item())

    @classmethod
    def as_cleared(cls, in_encoder: Encoder[Token, str], out_encoder: Encoder[Token, str]) -> Self:
        return cls(zeros(in_encoder.size, out_encoder.size, dtype=int32), in_encoder, out_encoder)
