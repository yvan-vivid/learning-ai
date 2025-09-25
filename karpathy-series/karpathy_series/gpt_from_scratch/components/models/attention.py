from typing import override

from torch import Tensor, arange
from torch.nn import Embedding, Linear

from karpathy_series.gpt_from_scratch.components.network.attention_block import AttentionBlock
from karpathy_series.gpt_from_scratch.components.network.model import ModelWithCrossEntropyLoss, WindowedGenerable

# Attention Model


class AttentionModel(ModelWithCrossEntropyLoss, WindowedGenerable):
    window_length: int
    code_size: int
    embedding_dims: int
    embedding: Embedding
    positions: Embedding
    block: AttentionBlock
    block2: AttentionBlock
    output: Linear

    def __init__(
        self, code_size: int, window_length: int, embedding_dims: int, head_count: int, head_size: int
    ) -> None:
        super().__init__()
        self.code_size = code_size
        self.window_length = window_length
        self.embedding = Embedding(code_size, embedding_dims)
        self.positions = Embedding(window_length, embedding_dims)
        self.block = AttentionBlock(embedding_dims, head_count, head_size)
        self.block2 = AttentionBlock(embedding_dims, head_count, head_size)
        self.output = Linear(embedding_dims, code_size)

    @override
    def forward(self, ix: Tensor) -> Tensor:
        window_length = ix.shape[-1]
        assert window_length == self.window_length, f"Mismatch ix is {ix.shape} vs window_length = {self.window_length}"

        embedded = self.embedding.forward(ix)
        positions = self.embedding.forward(arange(window_length))
        p_encoded = embedded * positions
        attention = self.block.forward(p_encoded)
        attention = self.block2.forward(attention)
        return self.output.forward(attention)
