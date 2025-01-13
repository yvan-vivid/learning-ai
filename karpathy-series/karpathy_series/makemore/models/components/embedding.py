from typing import Optional, override

from torch import Generator, Tensor, randn

from karpathy_series.makemore.models.components.component import BaseComponent


class Embedding(BaseComponent):
    embedding: Tensor

    def __init__(
        self,
        input_size: int,
        embedding_dims: int,
        init_scale: float = 1.0,
        generator: Optional[Generator] = None,
    ) -> None:
        init_factor: float = init_scale * input_size**-0.5
        self.embedding = (randn(input_size, embedding_dims, generator=generator) * init_factor).requires_grad_()

    @override
    def forward(self, x: Tensor, training: bool = False) -> Tensor:
        return self.embedding[x]

    @override
    def parameters(self) -> list[Tensor]:
        return [self.embedding]

    @override
    def describe(self) -> str:
        return f"Embedding [{self.embedding.shape[0]}, {self.embedding.shape[1]}]"
