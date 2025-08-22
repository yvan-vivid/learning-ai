from dataclasses import dataclass, replace
from typing import cast, override

from torch import Tensor

from karpathy_series.makemore.components.neuro.component import BaseComponent
from karpathy_series.makemore.components.typing import ArrayType, Dims


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
        dim = len(x.shape) + self.dim if self.dim < 0 else self.dim
        expanded = cast(Tensor, x.unflatten(dim, (-1, self.width)))  # type: ignore[no-untyped-call]
        if x.ndim > dim + 1:
            return expanded.flatten(dim + 1, dim + 2)
        else:
            return expanded

    @override
    def parameters(self) -> list[Tensor]:
        return []

    @override
    def describe(self) -> str:
        return f"Move a factor of {self.width} from {self.dim} to {self.dim + 1}"

    @override
    def type_transform(self, x: ArrayType) -> ArrayType:
        return replace(x, shape=x.shape.slide(self.dim, Dims(self.width)))
