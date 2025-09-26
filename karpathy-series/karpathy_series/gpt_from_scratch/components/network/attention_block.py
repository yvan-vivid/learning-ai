from typing import override

from torch import Tensor
from torch.nn import LayerNorm, Module

from karpathy_series.gpt_from_scratch.components.network.attention import MultiHeadSelfAttention
from karpathy_series.gpt_from_scratch.components.network.feed_forward import FeedForward


class AttentionBlock(Module):
    attention: MultiHeadSelfAttention
    feed_forward: FeedForward
    attention_norm: LayerNorm
    feed_forward_norm: LayerNorm

    def __init__(self, embedding_dims: int, head_count: int, head_size: int) -> None:
        super().__init__()

        self.attention_norm = LayerNorm(embedding_dims)
        self.attention = MultiHeadSelfAttention(embedding_dims, head_count, head_size)
        self.feed_forward_norm = LayerNorm(embedding_dims)
        self.feed_forward = FeedForward(embedding_dims, embedding_dims)

    @override
    def forward(self, x: Tensor) -> Tensor:
        attention = x + self.attention.forward(x)
        return x + self.feed_forward.forward(attention)
