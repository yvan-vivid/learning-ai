from typing import override

from torch import Tensor, arange
from torch.nn import Embedding, Module

# Attention Model


class PositionalEmbedding(Module):
    window_length: int
    code_size: int
    embedding_dims: int
    embedding: Embedding
    positions: Embedding

    def __init__(self, code_size: int, window_length: int, embedding_dims: int) -> None:
        super().__init__()
        self.code_size = code_size
        self.window_length = window_length
        self.embedding_dims = embedding_dims
        self.embedding = Embedding(code_size, embedding_dims)
        self.positions = Embedding(window_length, embedding_dims)

    @override
    def forward(self, ix: Tensor) -> Tensor:
        window_length = ix.shape[-1]
        assert window_length == self.window_length, f"Mismatch ix is {ix.shape} vs window_length = {self.window_length}"

        embedded = self.embedding.forward(ix)
        positions = self.embedding.forward(arange(window_length))
        return embedded * positions
