from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from itertools import product
from typing import Self, override

from torch import Tensor, int32, zeros

from karpathy_series.makemore.components.neuro.component import BaseComponent, GenerableComponent
from karpathy_series.makemore.components.typing import ArrayType
from karpathy_series.makemore.encoding.character import Token
from karpathy_series.makemore.util import norm_distro, sample_index_model


@dataclass(frozen=True)
class Frequentist(GenerableComponent, BaseComponent):
    counts: Tensor

    @override
    def forward(self, x: Tensor, training: bool = False) -> Tensor:
        return self.counts[x].float()

    @override
    def parameters(self) -> Iterable[Tensor]:
        return [self.counts]

    @override
    def describe(self) -> str:
        return "A frequentist model"

    @override
    def realize(self, u: Tensor) -> Token:
        return sample_index_model(norm_distro(u, -1))

    def items(self) -> Iterator[tuple[Token, Token, int]]:
        for i, j in product(range(self.counts.shape[0]), range(self.counts.shape[1])):
            yield i, j, int(self.counts[i, j].item())

    @override
    def type_transform(self, x: ArrayType) -> ArrayType:
        raise NotImplementedError

    @classmethod
    def as_cleared(cls, encoding_size: int, regularization: int = 0) -> Self:
        return cls(zeros(encoding_size, encoding_size, dtype=int32) + regularization)
