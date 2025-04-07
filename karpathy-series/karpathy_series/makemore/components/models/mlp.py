from dataclasses import dataclass
from functools import partial
from itertools import chain
from typing import Optional, Self, override

from torch import Generator

from karpathy_series.makemore.components.models.model import SequentialModel
from karpathy_series.makemore.components.neuro.batch_norm import BatchNorm1d
from karpathy_series.makemore.components.neuro.component import Component
from karpathy_series.makemore.components.neuro.embedding import Embedding
from karpathy_series.makemore.components.neuro.flatten import Flatten
from karpathy_series.makemore.components.neuro.functional import Tanh
from karpathy_series.makemore.components.neuro.linear import Linear
from karpathy_series.makemore.components.neuro.sequence import Sequence


@dataclass(frozen=True)
class MPLNet(SequentialModel):
    layers: Sequence
    encoding_size: int
    embedding_dims: int
    context_size: int

    @override
    def describe(self) -> str:
        return "An MLP model"

    @staticmethod
    def module(fan_in: int, fan_out: int, init_scale: float, generator: Optional[Generator]) -> tuple[Component, ...]:
        return (Linear(fan_in, fan_out, init_scale=init_scale, generator=generator), BatchNorm1d(fan_out), Tanh())

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
        module = partial(cls.module, init_scale=init_scale, generator=generator)
        output_layer = (
            (
                Linear(hidden_dims, encoding_size, generator=generator),
                BatchNorm1d(encoding_size, init_scale=0.01),
            )
            if norm_final_layer
            else (Linear(hidden_dims, encoding_size, init_scale=0.01, generator=generator),)
        )

        layers = Sequence(
            list(
                chain(
                    (Embedding(encoding_size, embedding_dims, generator=generator), Flatten(2)),
                    module(context_length, hidden_dims),
                    chain.from_iterable(module(hidden_dims, hidden_dims) for _ in range(num_layers - 1)),
                    output_layer,
                )
            )
        )

        return cls(
            layers,
            encoding_size,
            embedding_dims,
            context_size,
        )
