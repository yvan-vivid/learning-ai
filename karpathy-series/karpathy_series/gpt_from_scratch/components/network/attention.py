import math
from typing import override

from torch import Tensor, cat
from torch.nn import Linear, Module, ModuleList

from karpathy_series.gpt_from_scratch.util import masked_window


class SelfAttentionHead(Module):
    embedding_dims: int
    head_size: int
    query: Linear
    key: Linear
    value: Linear

    def __init__(self, embedding_dims: int, head_size: int) -> None:
        super().__init__()
        self.embedding_dims = embedding_dims
        self.head_size = head_size
        self.query = Linear(embedding_dims, head_size, bias=False)
        self.key = Linear(embedding_dims, head_size, bias=False)
        self.value = Linear(embedding_dims, head_size, bias=False)

    @override
    def forward(self, x: Tensor) -> Tensor:
        w = self.query.forward(x) @ self.key.forward(x).transpose(-2, -1)
        v = self.value.forward(x)
        return masked_window(w) @ v * math.pow(self.head_size, -0.5)


class MultiHeadSelfAttention(Module):
    embedding_dims: int
    head_count: int
    head_size: int
    heads: ModuleList

    def __init__(self, embedding_dims: int, head_count: int, head_size: int) -> None:
        super().__init__()
        self.embedding_dims = embedding_dims
        self.head_count = head_count
        self.head_size = head_size
        self.heads = ModuleList(SelfAttentionHead(embedding_dims, head_size) for _ in range(head_count))

    @override
    def forward(self, x: Tensor) -> Tensor:
        return cat([m.forward(x) for m in self.heads], dim=-1)
