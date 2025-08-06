import math
from dataclasses import dataclass
from typing import Self, override

from torch import Generator, Tensor, no_grad, ones, randn, tanh, zeros

from karpathy_series.makemore.components.neuro.component import BaseComponent, LogitGenerableComponent


@dataclass(frozen=True)
class MLPNet(LogitGenerableComponent, BaseComponent):
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
    def forward(self, x: Tensor, training: bool = False) -> Tensor:
        context_length = self.hidden_net.shape[0]
        w = self.input_net[x].view(-1, context_length)
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
    def describe(self) -> str:
        return "An MLP model built without components"

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
        generator: Generator | None,
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
        generator: Generator | None,
    ) -> Self:
        context_length = context_size * embedding_dims
        hidden_factor = (5.0 / 3.0) * math.pow(float(context_length), -0.5)
        return cls.init(
            randn(encoding_size, embedding_dims, generator=generator),
            randn(context_length, hidden_dims, generator=generator) * hidden_factor,
            randn(hidden_dims, encoding_size, generator=generator) * 0.01,
            randn(encoding_size, generator=generator) * 0.001,
            context_size,
        )
