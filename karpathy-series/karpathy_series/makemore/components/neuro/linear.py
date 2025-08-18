import math
from typing import Optional, override

from torch import Generator, Tensor, randn, zeros

from karpathy_series.makemore.components.neuro.component import BaseComponent


class Linear(BaseComponent):
    fan_in: int
    fan_out: int
    weight: Tensor
    bias: Optional[Tensor] = None

    def __init__(
        self,
        fan_in: int,
        fan_out: int,
        bias: bool = True,
        init_scale: float = 1.0,
        generator: Generator | None = None,
    ) -> None:
        self.fan_in = fan_in
        self.fan_out = fan_out
        init_factor = init_scale * math.pow(fan_in, -0.5)
        self.weight = (randn(fan_in, fan_out, generator=generator) * init_factor).requires_grad_()
        if bias:
            self.bias = zeros(fan_out).requires_grad_()

    @override
    def forward(self, x: Tensor, training: bool = False) -> Tensor:
        out = x @ self.weight
        if (b := self.bias) is not None:
            out += b
        return out

    @override
    def parameters(self) -> list[Tensor]:
        return [self.weight] + ([] if self.bias is None else [self.bias])

    @override
    def describe(self) -> str:
        suffix = "" if self.bias is None else " with bias"
        return f"Linear [{self.fan_in}, {self.fan_out}]{suffix}"

    @override
    def shape(self, x: tuple[int, ...]) -> tuple[int, ...]:
        assert x[-1] == self.fan_in, f"input shape {x} incompatible with fan-in {self.fan_in}"
        return (*x[:-1], self.fan_out)
