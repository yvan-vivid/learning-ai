from dataclasses import dataclass
from typing import Self, override

from torch import Tensor, randn, tanh

from karpathy_series.makemore.components.neuro.component import BaseComponent
from karpathy_series.makemore.util import sliding_window


@dataclass(frozen=True)
class Perceptron(BaseComponent):
    input_net: Tensor
    hidden_layers: list[Tensor]

    @override
    def describe(self) -> str:
        return "A simple perceptron model"

    @override
    def parameters(self) -> list[Tensor]:
        return [self.input_net] + self.hidden_layers

    @override
    def forward(self, x: Tensor, training: bool = False) -> Tensor:
        m = tanh(self.input_net[x])
        for wa in self.hidden_layers:
            m = tanh(m @ wa)
        return m

    @classmethod
    def init_random_from_size(cls, in_size: int, out_size: int, hidden: list[int]) -> Self:
        hidden.append(out_size)
        return cls(
            randn(in_size, hidden[0], requires_grad=True),
            [randn(p, requires_grad=True) for p in sliding_window(hidden, 2)],
        )
