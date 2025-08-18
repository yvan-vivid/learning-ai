from dataclasses import dataclass
from functools import partial
from typing import Optional, Self, cast, override

from torch import Generator

from karpathy_series.makemore.components.neuro.batch_norm import BatchNorm1d
from karpathy_series.makemore.components.neuro.component import Component, LogitGenerableComponent
from karpathy_series.makemore.components.neuro.embedding import Embedding
from karpathy_series.makemore.components.neuro.flatten import Flatten
from karpathy_series.makemore.components.neuro.functional import Tanh
from karpathy_series.makemore.components.neuro.linear import Linear
from karpathy_series.makemore.components.neuro.sequence import Sequence
from karpathy_series.makemore.components.neuro.slide import Slide


@dataclass(frozen=True)
class Wavenet(LogitGenerableComponent, Sequence):
    encoding_size: int
    embedding_dims: int

    @override
    def describe(self) -> str:
        parts = super().describe()
        return f"A wavenet model: {parts}"

    @staticmethod
    def module(fan_in: int, fan_out: int, init_scale: float, generator: Optional[Generator]) -> tuple[Component, ...]:
        return (
            Linear(fan_in, fan_out, init_scale=init_scale, generator=generator),
            BatchNorm1d(fan_out),
            Tanh(),
        )

    @classmethod
    def init(
        cls,
        levels: int,
        reduction_factor: int,
        encoding_size: int,
        embedding_dims: int,
        context_size: int,
        hidden_dims: int,
        # norm_final_layer: bool = True,
        generator: Optional[Generator] = None,
    ) -> Self:
        reductions = max(levels - 1, 1)
        total_reduction = cast(int, reduction_factor**reductions)
        final_dims = context_size // total_reduction
        if total_reduction * final_dims != context_size:
            raise ValueError(f"{context_size} does not factor out {reduction_factor} for {levels} layers")

        assert levels == 4

        init_scale = 5.0 / 3.0
        module = partial(cls.module, init_scale=init_scale, generator=generator)
        layers = [
            # N[b, f^3 n] => R[b, f^3 n, d]
            Embedding(encoding_size, embedding_dims),
            # R[b, f^3 n, d] => R[b, f^2 n, fd]
            Slide(-2, reduction_factor),
            # R[b, f^2 n, fd] => R[b, f^2 n, h]
            *module(reduction_factor * embedding_dims, hidden_dims),
            # R[b, f^2 n, h] => R[b, fn, fh]
            Slide(-2, reduction_factor),
            # R[b, fn, fh] => R[b, fn, h]
            *module(reduction_factor * hidden_dims, hidden_dims),
            # R[b, fn, h] => R[b, n, fh]
            Slide(-2, reduction_factor),
            # R[b, n, fh] => R[b, n, h]
            *module(reduction_factor * hidden_dims, hidden_dims),
            # R[b, n, h] => R[b, nh]
            Flatten(2),
            # R[b, nh] => R[b, N]
            Linear(final_dims * hidden_dims, encoding_size),
            # BatchNorm1d(embedding_dims, init_scale=0.01),
        ]

        return cls(layers, encoding_size, embedding_dims)
