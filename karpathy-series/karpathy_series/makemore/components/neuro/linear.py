import math
from typing import Optional, override

from torch import Generator, Tensor, randn, zeros

from karpathy_series.makemore.components.neuro.component import BaseComponent
from karpathy_series.makemore.components.typing import ArrayType, ArrayTypeError, Dims, TensorType


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
    def type_transform(self, x: ArrayType) -> ArrayType:
        if not isinstance(x, TensorType):
            raise ArrayTypeError(f"Input type {x} not scalar")

        if (active := x.shape.active()) is None or self.fan_in != active.value:
            raise ArrayTypeError(f"Active input of {x} doesn't match fan-in {self.fan_in}")

        return x.replace_active(Dims(self.fan_out))
