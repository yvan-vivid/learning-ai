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
    def shape(self, x: tuple[int, ...]) -> tuple[int, ...]:
        dim = len(x) + self.dim if self.dim < 0 else self.dim
        assert 0 <= dim < len(x), f"'dim' {dim} out of bounds for size {x}"
        w = self.width
        xd = x[dim]
        f = xd // w
        assert xd == w * f, f"{w} not a factor of {xd} at {dim} in {x}"
        suffix = (w * x[dim + 1], *x[dim + 2 :]) if dim < len(x) - 1 else (w,)
        return (*x[:dim], f, *suffix)
