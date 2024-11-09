from dataclasses import dataclass
from functools import partial
from itertools import chain, repeat, starmap
from typing import Optional, Self, override

from torch import Generator, Tensor

from karpathy_series.makemore.models.components.batch_norm import BatchNorm1d
from karpathy_series.makemore.models.components.component import Component, ComponentRecorder
from karpathy_series.makemore.models.components.embedding import Embedding
from karpathy_series.makemore.models.components.flatten import Flatten
from karpathy_series.makemore.models.components.functional import Tanh
from karpathy_series.makemore.models.components.linear import Linear
from karpathy_series.makemore.models.components.sequence import Sequence
from karpathy_series.makemore.util import sliding_window

from .sequential import SequentialNet


@dataclass(frozen=True)
class MPLNet(SequentialNet):
    layers: Sequence
    encoding_size: int
    embedding_dims: int
    context_size: int

    @override
    def parameters(self) -> list[Tensor]:
        return list(self.layers.parameters())

    @override
    def forward(self, xis: Tensor, training: bool = False, record: Optional[ComponentRecorder] = None) -> Tensor:
        return self.layers(xis, training, record)

    @staticmethod
    def module(fan_in: int, fan_out: int, init_scale: float, generator: Optional[Generator]) -> list[Component]:
        return [Linear(fan_in, fan_out, init_scale=init_scale, generator=generator), BatchNorm1d(fan_out), Tanh()]

    @classmethod
    def init(
        cls,
        num_layers: int,
        encoding_size: int,
        embedding_dims: int,
        context_size: int,
        hidden_dims: int,
        generator: Optional[Generator],
    ) -> Self:
        init_scale = 5.0 / 3.0
        context_length = embedding_dims * context_size
        boundaries = sliding_window(chain((context_length,), repeat(hidden_dims, num_layers)), 2)
        module = partial(cls.module, init_scale=init_scale, generator=generator)
        layers = Sequence(
            list(
                chain(
                    (Embedding(encoding_size, embedding_dims, generator=generator), Flatten(2)),
                    chain.from_iterable(starmap(module, boundaries)),
                    (
                        Linear(hidden_dims, encoding_size, init_scale=0.01, generator=generator),
                        BatchNorm1d(encoding_size),
                    ),
                )
            )
        )

        return cls(
            layers,
            encoding_size,
            embedding_dims,
            context_size,
        )
