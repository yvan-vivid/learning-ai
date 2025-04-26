from dataclasses import dataclass
from itertools import product
from typing import Iterator, Self

from torch import Tensor, int32, zeros

from karpathy_series.makemore.components.function.loss import CrossEntropyExpLoss, Loss
from karpathy_series.makemore.components.neuro.component import Component, ComponentRecording
from karpathy_series.makemore.encoding.character import Token
from karpathy_series.makemore.util import norm_distro, sample_index_logits, sample_index_model


@dataclass(frozen=True)
class NetModel:
    component: Component
    loss: Loss

    def run(self, x: Tensor, y: Tensor, training: bool = True, record: ComponentRecording = None) -> Tensor:
        """N[s] x M[s] => L"""
        return self.loss(self.component(x, training=training, record=record), y)

    def backward(self, loss: Tensor) -> None:
        for wa in self.component.parameters():
            wa.grad = None
        loss.backward()  # type: ignore[no-untyped-call]

    def generate(self, x: Tensor) -> Token:
        return sample_index_logits(self.component(x))


@dataclass(frozen=True)
class FreqModel:
    encoding_size: int
    counts: Tensor
    loss: Loss

    def forward(self, x: Tensor) -> Tensor:
        return self.counts[x].float()

    def run(self, x: Tensor, y: Tensor) -> Tensor:
        """N[s] x M[s] => L"""
        return self.loss(self.forward(x), y)

    def generate(self, x: Tensor) -> Token:
        return sample_index_model(norm_distro(self.forward(x), -1))

    def items(self) -> Iterator[tuple[Token, Token, int]]:
        for i, j in product(range(self.encoding_size), range(self.encoding_size)):
            yield i, j, int(self.counts[i, j].item())

    @classmethod
    def as_cleared(cls, encoding_size: int, regularization: int = 0) -> Self:
        return cls(
            encoding_size, zeros(encoding_size, encoding_size, dtype=int32) + regularization, CrossEntropyExpLoss()
        )
