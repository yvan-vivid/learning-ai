from dataclasses import dataclass, replace
from typing import cast, override

from torch import Tensor

from karpathy_series.makemore.components.neuro.component import BaseComponent
from karpathy_series.makemore.components.typing import ArrayType, Dims


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
    def type_transform(self, x: ArrayType) -> ArrayType:
        return replace(x, shape=x.shape.factor(self.dim, Dims(self.width)))
