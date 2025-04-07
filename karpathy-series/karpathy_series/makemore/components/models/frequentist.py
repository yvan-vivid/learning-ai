from collections.abc import Iterator
from dataclasses import dataclass
from itertools import product
from typing import Self, override

from torch import Tensor, int32, zeros

from karpathy_series.makemore.components.models.model import Model
from karpathy_series.makemore.components.neuro.component import ComponentRecording
from karpathy_series.makemore.encoding.character import Token
from karpathy_series.makemore.util import norm_distro, sample_index_model


@dataclass(frozen=True)
class FreqModel(Model):
    encoding_size: int
    counts: Tensor

    @override
    def describe(self) -> str:
        return "A frequency table model"

    @override
    def parameters(self) -> list[Tensor]:
        return [self.counts]

    @override
    def __call__(self, x: Tensor, training: bool = False, recorder: ComponentRecording = None) -> Tensor:
        return self.counts[x].float()

    @override
    def generate(self, x: Tensor) -> Token:
        return sample_index_model(norm_distro(self(x), -1))

    def train(self, xis: Tensor, yis: Tensor) -> None:
        assert xis.shape == yis.shape
        for xi, yi in zip(list(xis), list(yis)):
            self.counts[xi, yi] += 1

    def items(self) -> Iterator[tuple[Token, Token, int]]:
        for i, j in product(range(self.encoding_size), range(self.encoding_size)):
            yield i, j, int(self.counts[i, j].item())

    @classmethod
    def as_cleared(cls, encoding_size: int, regularization: int = 0) -> Self:
        return cls(encoding_size, zeros(encoding_size, encoding_size, dtype=int32) + regularization)
