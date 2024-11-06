from dataclasses import dataclass
from functools import partial
from itertools import chain, repeat, starmap
from typing import Optional, Self, override

from torch import Generator, Tensor, no_grad, ones, randn, tanh, zeros

from karpathy_series.makemore.models.components.batch_norm import BatchNorm1d
from karpathy_series.makemore.models.components.component import Component
from karpathy_series.makemore.models.components.embedding import Embedding
from karpathy_series.makemore.models.components.flatten import Flatten
from karpathy_series.makemore.models.components.functional import Tanh
from karpathy_series.makemore.models.components.linear import Linear
from karpathy_series.makemore.util import sliding_window

from .sequential import CalcRecorder, SequentialNet


@dataclass(frozen=True)
class MPLNet(SequentialNet):
    input_net: Tensor
    hidden_net: Tensor
    hidden_gain: Tensor
    hidden_bias: Tensor
    output_net: Tensor
    output_bias: Tensor
    context_size: int

    batch_mean: Tensor
    batch_std: Tensor

    @override
    def parameters(self) -> list[Tensor]:
        return [
            self.input_net,
            self.hidden_net,
            self.hidden_gain,
            self.hidden_bias,
            self.output_net,
            self.output_bias,
        ]

    @override
    def forward(self, xis: Tensor, training: bool = False, record: Optional[CalcRecorder] = None) -> Tensor:
        context_length = self.hidden_net.shape[0]
        w = self.input_net[xis].view(-1, context_length)
        v = w @ self.hidden_net
        if not training:
            v_norm = (v - self.batch_mean) / self.batch_std
            h = tanh(self.hidden_gain * v_norm + self.hidden_bias)
            return h @ self.output_net + self.output_bias

        hidden_mean = v.mean(0, keepdim=True)
        hidden_std = v.std(0, keepdim=True)
        v_norm = (v - hidden_mean) / hidden_std

        with no_grad():
            self.batch_mean.data = 0.999 * self.batch_mean + 0.001 * hidden_mean
            self.batch_std.data = 0.999 * self.batch_std + 0.001 * hidden_std

        h = tanh(self.hidden_gain * v_norm + self.hidden_bias)
        return h @ self.output_net + self.output_bias

    @classmethod
    def init(
        cls,
        input_net: Tensor,
        hidden_net: Tensor,
        output_net: Tensor,
        output_bias: Tensor,
        context_size: int,
    ) -> Self:
        return cls(
            input_net.requires_grad_(),
            hidden_net.requires_grad_(),
            ones(1, hidden_net.shape[-1]).requires_grad_(),
            zeros(1, hidden_net.shape[-1]).requires_grad_(),
            output_net.requires_grad_(),
            output_bias.requires_grad_(),
            context_size,
            zeros(1, hidden_net.shape[-1]),
            ones(1, hidden_net.shape[-1]),
        )

    @classmethod
    def init_random_from_size(
        cls,
        encoding_size: int,
        context_size: int,
        embedding_dims: int,
        hidden_dims: int,
        generator: Optional[Generator],
    ) -> Self:
        context_length = context_size * embedding_dims
        return cls.init(
            randn(encoding_size, embedding_dims, generator=generator),
            randn(context_length, hidden_dims, generator=generator),
            randn(hidden_dims, encoding_size, generator=generator),
            randn(encoding_size, generator=generator),
            context_size,
        )

    @classmethod
    def init_normalized_from_size(
        cls,
        encoding_size: int,
        context_size: int,
        embedding_dims: int,
        hidden_dims: int,
        generator: Optional[Generator],
    ) -> Self:
        context_length = context_size * embedding_dims
        hidden_factor: float = (5.0 / 3.0) * (float(context_length) ** -0.5)
        return cls.init(
            randn(encoding_size, embedding_dims, generator=generator),
            randn(context_length, hidden_dims, generator=generator) * hidden_factor,
            randn(hidden_dims, encoding_size, generator=generator) * 0.01,
            randn(encoding_size, generator=generator) * 0.001,
            context_size,
        )


@dataclass(frozen=True)
class MPLNetComponents(SequentialNet):
    layers: list[Component]
    encoding_size: int
    embedding_dims: int
    context_size: int

    @override
    def parameters(self) -> list[Tensor]:
        return [p for c in self.layers for p in c.parameters()]

    @override
    def forward(self, xis: Tensor, training: bool = False, record: Optional[CalcRecorder] = None) -> Tensor:
        out = xis
        for layer in self.layers:
            out = layer(out, training)
            if record is not None:
                record.record(layer, out)
                out.retain_grad()
        return out

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
        layers: list[Component] = list(
            chain(
                (Embedding(encoding_size, embedding_dims, generator=generator), Flatten(2)),
                chain.from_iterable(starmap(module, boundaries)),
                (Linear(hidden_dims, encoding_size, init_scale=0.01, generator=generator), BatchNorm1d(encoding_size)),
            )
        )

        return cls(
            layers,
            encoding_size,
            embedding_dims,
            context_size,
        )
