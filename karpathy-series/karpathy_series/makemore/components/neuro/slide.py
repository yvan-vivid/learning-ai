from dataclasses import dataclass
from typing import override

from torch import Tensor

from karpathy_series.makemore.components.neuro.component import BaseComponent


@dataclass(frozen=True)
class Slide(BaseComponent):
    """
    `Slide(d, w)` does an `Expand(d, w)` followed by `Flatten(d+1)`
    """

    dim: int
    width: int

    @override
    def forward(self, x: Tensor, training: bool = False) -> Tensor:
        expanded = x.unflatten(self.dim, (-1, self.width))  # type: ignore[no-untyped-call]
        if x.ndim > self.dim + 1:
            return expanded.flatten(self.dim + 1, self.dim + 2)  # type: ignore[no-any-return]
        else:
            return expanded  # type: ignore[no-any-return]

    @override
    def parameters(self) -> list[Tensor]:
        return []

    @override
    def describe(self) -> str:
        return f"Move a factor of {self.width} from {self.dim} to {self.dim + 1}"
