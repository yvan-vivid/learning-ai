from dataclasses import dataclass
from functools import partial
from typing import Callable, override

from torch import Tensor, tanh

from karpathy_series.makemore.components.neuro.component import BaseComponent
from karpathy_series.makemore.components.typing import ArrayType


@dataclass(frozen=True)
class Functional(BaseComponent):
    fun: Callable[[Tensor], Tensor]

    @override
    def forward(self, x: Tensor, training: bool = False) -> Tensor:
        return self.fun(x)

    @override
    def parameters(self) -> list[Tensor]:
        return []

    @override
    def describe(self) -> str:
        return f"Functional {self.fun}"

    @override
    def type_transform(self, x: ArrayType) -> ArrayType:
        return x


Tanh = partial(Functional, tanh)
