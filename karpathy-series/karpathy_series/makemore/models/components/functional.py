from dataclasses import dataclass
from functools import partial
from typing import Callable, override

from torch import Tensor, tanh

from karpathy_series.makemore.models.components.component import Component


@dataclass(frozen=True)
class Functional(Component):
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


Tanh = partial(Functional, tanh)
