from typing import override

from torch import Tensor
from torch.nn import Linear, Module, ReLU


class FeedForward(Module):
    fan_in: int
    fan_out: int
    linear: Linear
    activation: ReLU

    def __init__(self, fan_in: int, fan_out: int) -> None:
        super().__init__()
        self.fan_in = fan_in
        self.fan_out = fan_out
        self.linear = Linear(fan_in, fan_out)
        self.activation = ReLU()

    @override
    def forward(self, x: Tensor) -> Tensor:
        return self.activation.forward(self.linear.forward(x))
