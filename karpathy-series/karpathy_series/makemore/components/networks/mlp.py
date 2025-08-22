from dataclasses import dataclass
from typing import Optional, Self, override

from torch import Generator

from karpathy_series.makemore.components.networks.blocks import LinearActivationBlocks, LinearOutputBlock
from karpathy_series.makemore.components.neuro.component import Component, LogitGenerableComponent
from karpathy_series.makemore.components.neuro.embedding import Embedding
from karpathy_series.makemore.components.neuro.flatten import Flatten
from karpathy_series.makemore.components.neuro.sequence import Sequence


@dataclass(frozen=True)
class MPLNet(LogitGenerableComponent, Sequence):
    encoding_size: int
    embedding_dims: int

    @override
    def describe(self) -> str:
        parts = super().describe()
        return f"An MLP model: {parts}"

    @classmethod
    def init(
        cls,
        num_layers: int,
        encoding_size: int,
        embedding_dims: int,
        context_size: int,
        hidden_dims: int,
        norm_final_layer: bool = True,
        generator: Optional[Generator] = None,
    ) -> Self:
        init_scale = 5.0 / 3.0
        context_length = embedding_dims * context_size
        structure = (context_length, *((hidden_dims,) * (num_layers - 1)))
        layers: list[Component] = [
            Embedding(encoding_size, embedding_dims, generator=generator),
            Flatten(2),
            LinearActivationBlocks.init(structure, init_scale, generator=generator),
            LinearOutputBlock.init(
                hidden_dims, encoding_size, init_scale, normalize=norm_final_layer, generator=generator
            ),
        ]
        return cls(layers, encoding_size, embedding_dims)
