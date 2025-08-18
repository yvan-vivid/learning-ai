from dataclasses import dataclass
from functools import partial
from typing import Callable, override

from torch import Tensor, tanh

from karpathy_series.makemore.components.neuro.component import BaseComponent


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
    def shape(self, x: tuple[int, ...]) -> tuple[int, ...]:
        return x


Tanh = partial(Functional, tanh)
