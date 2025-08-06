import math
from typing import override

from torch import Generator, Tensor, randn

from karpathy_series.makemore.components.neuro.component import BaseComponent


class Embedding(BaseComponent):
    """
    Given a discrete input space of `N` this embeds into `R[d]`, where `d` is the target dimension.
    """

    embedding: Tensor

    def __init__(
        self,
        input_size: int,
        embedding_dims: int,
        init_scale: float = 1.0,
        generator: Generator | None = None,
    ) -> None:
        init_factor = init_scale * math.pow(input_size, -0.5)
        self.embedding = (randn(input_size, embedding_dims, generator=generator) * init_factor).requires_grad_()

    @override
    def forward(self, x: Tensor, training: bool = False) -> Tensor:
        """N[s] -> R[s, d]"""
        return self.embedding[x]

    @override
    def parameters(self) -> list[Tensor]:
        return [self.embedding]

    @override
    def describe(self) -> str:
        return f"Embedding [{self.embedding.shape[0]}, {self.embedding.shape[1]}]"
