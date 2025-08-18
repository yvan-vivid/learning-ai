from dataclasses import dataclass
from math import prod
from typing import override

from torch import Tensor

from karpathy_series.makemore.components.neuro.component import BaseComponent


@dataclass(frozen=True)
class Flatten(BaseComponent):
    """
    `Flatten(d)` flattens the `last` dimension of an array
    For instance, last = 1, will not flatten anything,
    while last = 3, will flatten the last 3 dimensions.
    """

    last: int

    @override
    def forward(self, x: Tensor, training: bool = False) -> Tensor:
        return x.flatten(x.ndim - self.last)

    @override
    def parameters(self) -> list[Tensor]:
        return []

    @override
    def describe(self) -> str:
        return f"Flatten last {self.last} dims"

    @override
    def shape(self, x: tuple[int, ...]) -> tuple[int, ...]:
        assert 0 < self.last <= len(x), f"input {x} not wide enough for last = {self.last}"
        k = len(x) - self.last
        return (*x[:k], prod(x[k:]))
