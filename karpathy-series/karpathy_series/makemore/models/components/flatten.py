from dataclasses import dataclass
from typing import override

from torch import Tensor

from karpathy_series.makemore.models.components.component import BaseComponent


@dataclass(frozen=True)
class Flatten(BaseComponent):
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
