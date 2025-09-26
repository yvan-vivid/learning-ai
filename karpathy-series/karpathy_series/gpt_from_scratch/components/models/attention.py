from typing import cast, override

from torch import Tensor
from torch.nn import Linear, ModuleList

from karpathy_series.gpt_from_scratch.components.network.attention_block import AttentionBlock
from karpathy_series.gpt_from_scratch.components.network.model import ModelWithCrossEntropyLoss, WindowedGenerable
from karpathy_series.gpt_from_scratch.components.network.positional_embedding import PositionalEmbedding

# Attention Model


class AttentionModel(ModelWithCrossEntropyLoss, WindowedGenerable):
    window_length: int
    code_size: int
    embedding_dims: int
    embedding: PositionalEmbedding
    blocks: ModuleList
    output: Linear

    def __init__(
        self, code_size: int, window_length: int, embedding_dims: int, blocks: int, head_count: int, head_size: int
    ) -> None:
        super().__init__()
        self.code_size = code_size
        self.window_length = window_length
        self.embedding_dims = embedding_dims
        self.embedding = PositionalEmbedding(code_size, window_length, embedding_dims)
        self.blocks = ModuleList(AttentionBlock(embedding_dims, head_count, head_size) for _ in range(blocks))
        self.output = Linear(embedding_dims, code_size)

    @override
    def forward(self, ix: Tensor) -> Tensor:
        embedded = self.embedding.forward(ix)
        for m in self.blocks:
            embedded = cast(Tensor, m.forward(embedded))
        return self.output.forward(embedded)
