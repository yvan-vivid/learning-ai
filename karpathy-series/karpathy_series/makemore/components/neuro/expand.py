from dataclasses import dataclass
from typing import cast, override

from torch import Tensor

from karpathy_series.makemore.components.neuro.component import BaseComponent


@dataclass(frozen=True)
class Expand(BaseComponent):
    """
    `Expand(d, w)` expands the `d`th dimension of an array into batches of size `w`
    The shape will go from `(..., n_d, ...)` to `(..., b, w, ...)`
        where `n_d = b * w`
    If the constraint is not met, `forward` will throw an exception
    """

    dim: int
    width: int

    @override
    def forward(self, x: Tensor, training: bool = False) -> Tensor:
        return cast(Tensor, x.unflatten(self.dim, (-1, self.width)))  # type: ignore[no-untyped-call]

    @override
    def parameters(self) -> list[Tensor]:
        return []

    @override
    def describe(self) -> str:
        return f"Expand dim {self.dim} into {self.width} sized batches"

    @override
    def shape(self, x: tuple[int, ...]) -> tuple[int, ...]:
        dim = len(x) + self.dim if self.dim < 0 else self.dim
        assert 0 <= dim < len(x), f"dim {dim} out of bounds for {x}"
        w = self.width
        xd = x[dim]
        f = xd // w
        assert xd == w * f, f"{w} not a factor of {xd} at {dim} in {x}"
        return (*x[: dim - 1], f, w, *x[dim:])
