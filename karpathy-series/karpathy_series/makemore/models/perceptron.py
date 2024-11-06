from dataclasses import dataclass
from typing import Optional, Self, override

from torch import Tensor, randn, tanh

from karpathy_series.makemore.models.sequential import CalcRecorder, SequentialNet

from ..util import sliding_window


@dataclass(frozen=True)
class Perceptron(SequentialNet):
    input_net: Tensor
    hidden_layers: list[Tensor]

    @override
    def parameters(self) -> list[Tensor]:
        return [self.input_net] + self.hidden_layers

    @override
    def forward(self, xis: Tensor, training: bool = False, record: Optional[CalcRecorder] = None) -> Tensor:
        m = tanh(self.input_net[xis])
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
