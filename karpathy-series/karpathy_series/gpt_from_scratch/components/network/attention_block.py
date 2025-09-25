from typing import override

from torch import Tensor
from torch.nn import Module

from karpathy_series.gpt_from_scratch.components.network.attention import MultiHeadSelfAttention
from karpathy_series.gpt_from_scratch.components.network.feed_forward import FeedForward


class AttentionBlock(Module):
    attention: MultiHeadSelfAttention
    feed_forward: FeedForward

    def __init__(self, embedding_dims: int, head_count: int, head_size: int) -> None:
        super().__init__()
        self.attention = MultiHeadSelfAttention(embedding_dims, head_count, head_size)
        self.feed_forward = FeedForward(head_count * head_size, embedding_dims)

    @override
    def forward(self, x: Tensor) -> Tensor:
        attention = x + self.attention.forward(x)
        return x + self.feed_forward.forward(attention)
