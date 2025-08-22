from typing import override

from torch import Generator, Tensor, randn

from karpathy_series.makemore.components.neuro.component import BaseComponent, LogitGenerableComponent
from karpathy_series.makemore.components.typing import ArrayType


class LinearNetwork(LogitGenerableComponent, BaseComponent):
    """
    This is a model `m` with N ins and M outs. Supposing `s` is a batch shape:
        `x : N[s]`
        `m(x) : R[s, M]`
    The model does both a one-hot embedding and a matrix application directly
    through indexing.
    """

    """R[N, M]"""
    wa: Tensor

    @override
    def forward(self, x: Tensor, training: bool = False) -> Tensor:
        """N[s] => R[s, M]"""
        return self.wa[x]

    @override
    def parameters(self) -> list[Tensor]:
        return [self.wa]

    @override
    def describe(self) -> str:
        return f"Linear network: [{self.wa.shape[0]}, {self.wa.shape[1]}]"

    def __init__(
        self,
        fan_in: int,
        fan_out: int,
        generator: Generator | None = None,
    ) -> None:
        self.wa = randn(fan_in, fan_out, generator=generator).requires_grad_()

    @override
    def type_transform(self, x: ArrayType) -> ArrayType:
        raise NotImplementedError
