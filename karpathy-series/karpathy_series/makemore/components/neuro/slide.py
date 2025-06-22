from dataclasses import dataclass
from typing import cast, override

from torch import Tensor

from karpathy_series.makemore.components.neuro.component import BaseComponent


@dataclass(frozen=True)
class Slide(BaseComponent):
    """
    `Slide(d, w)` does an `Expand(d, w)` followed by a flattening of the dimension d+1 with d+2
    In terms of shapes,
        `(..., n*w, y, ...) => (..., n, y*w, ...)`
    with the rank preserved, excpet in the case that `d` is the last dimension, in which case
    this will just be an expansion of the last dimension.
    """

    dim: int
    width: int

    @override
    def forward(self, x: Tensor, training: bool = False) -> Tensor:
        expanded = cast(Tensor, x.unflatten(self.dim, (-1, self.width)))  # type: ignore[no-untyped-call]
        if x.ndim > self.dim + 1:
            return expanded.flatten(self.dim + 1, self.dim + 2)
        else:
            return expanded

    @override
    def parameters(self) -> list[Tensor]:
        return []

    @override
    def describe(self) -> str:
        return f"Move a factor of {self.width} from {self.dim} to {self.dim + 1}"
