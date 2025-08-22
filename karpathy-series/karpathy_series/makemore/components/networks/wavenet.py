from dataclasses import dataclass
from itertools import chain
from typing import Optional, Self, cast, override

from torch import Generator

from karpathy_series.makemore.components.networks.blocks import LinearActivationBlock, LinearOutputBlock
from karpathy_series.makemore.components.neuro.component import Component, LogitGenerableComponent
from karpathy_series.makemore.components.neuro.embedding import Embedding
from karpathy_series.makemore.components.neuro.flatten import Flatten
from karpathy_series.makemore.components.neuro.sequence import Sequence
from karpathy_series.makemore.components.neuro.slide import Slide
from karpathy_series.makemore.util import affine_sequence


@dataclass(frozen=True)
class Wavenet(LogitGenerableComponent, Sequence):
    encoding_size: int
    embedding_dims: int

    @override
    def describe(self) -> str:
        parts = super().describe()
        return f"A wavenet model: {parts}"

    @classmethod
    def init(
        cls,
        levels: int,
        reduction_factor: int,
        encoding_size: int,
        embedding_dims: int,
        context_size: int,
        hidden_dims: int,
        norm_final_layer: bool = True,
        generator: Optional[Generator] = None,
    ) -> Self:
        reductions = max(levels - 1, 1)
        total_reduction = cast(int, reduction_factor**reductions)
        final_dims = context_size // total_reduction
        if total_reduction * final_dims != context_size:
            raise ValueError(f"{context_size} does not factor out {reduction_factor} for {levels} layers")

        assert levels == 4
        init_scale = 5.0 / 3.0
        layers: list[Component] = [
            Embedding(encoding_size, embedding_dims),
            *chain.from_iterable(
                [
                    Slide(-2, reduction_factor),
                    LinearActivationBlock.init(
                        reduction_factor * dims, hidden_dims, init_scale=init_scale, generator=generator
                    ),
                ]
                for dims in affine_sequence(embedding_dims, hidden_dims, reductions - 1)
            ),
            Flatten(2),
            LinearOutputBlock.init(
                final_dims * hidden_dims,
                encoding_size,
                init_scale=0.01,
                normalize=norm_final_layer,
                generator=generator,
            ),
        ]

        return cls(layers, encoding_size, embedding_dims)
