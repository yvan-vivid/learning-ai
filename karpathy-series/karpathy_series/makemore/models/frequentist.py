from dataclasses import dataclass
from itertools import product
from typing import Iterator, Self, Tuple, override

from torch import Tensor, int32, zeros

from karpathy_series.makemore.models.sequential import SequentialNet

from ..encoding.character import Token
from ..util import cross_entropy_exp, norm_distro, sample_index_model


@dataclass(frozen=True)
class FreqModel(SequentialNet):
    encoding_size: int
    counts: Tensor

    @override
    def parameters(self) -> list[Tensor]:
        return [self.counts]

    @override
    def forward(self, xis: Tensor) -> Tensor:
        return self.counts[xis].float()

    @override
    def loss(self, u: Tensor, yis: Tensor) -> Tensor:
        return cross_entropy_exp(u, yis)

    @override
    def generate(self, xi: Tensor) -> Token:
        return sample_index_model(norm_distro(self.forward(xi), -1))

    def train(self, xis: Tensor, yis: Tensor) -> None:
        assert xis.shape == yis.shape
        for xi, yi in zip(list(xis), list(yis)):
            self.counts[xi, yi] += 1

    def items(self) -> Iterator[Tuple[Token, Token, int]]:
        for i, j in product(range(self.encoding_size), range(self.encoding_size)):
            yield i, j, int(self.counts[i, j].item())

    @classmethod
    def as_cleared(cls, encoding_size: int, regularization: int = 0) -> Self:
        return cls(encoding_size, zeros(encoding_size, encoding_size, dtype=int32) + regularization)
