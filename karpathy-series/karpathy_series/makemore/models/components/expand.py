from dataclasses import dataclass
from typing import override

from torch import Tensor

from karpathy_series.makemore.models.components.component import BaseComponent


@dataclass(frozen=True)
class Expand(BaseComponent):
    dim: int
    width: int

    @override
    def forward(self, x: Tensor, training: bool = False) -> Tensor:
        return x.unflatten(self.dim, (-1, self.width))  # type: ignore[no-untyped-call,no-any-return]

    @override
    def parameters(self) -> list[Tensor]:
        return []

    @override
    def describe(self) -> str:
        return f"Expand dim {self.dim} into {self.width} sized batches"
