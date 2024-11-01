from dataclasses import dataclass
from typing import Optional, Self, override

from torch import Generator, Tensor, randn, tanh

from .sequential import SequentialNet


@dataclass(frozen=True)
class MPLNet(SequentialNet):
    input_net: Tensor
    hidden_net: Tensor
    hidden_bias: Tensor
    output_net: Tensor
    output_bias: Tensor
    context_size: int

    @override
    def parameters(self) -> list[Tensor]:
        return [self.input_net, self.hidden_net, self.hidden_bias, self.output_net, self.output_bias]

    @override
    def forward(self, xis: Tensor) -> Tensor:
        context_length = self.hidden_net.shape[0]
        w = self.input_net[xis].view(-1, context_length)
        h = tanh(w @ self.hidden_net + self.hidden_bias)
        return h @ self.output_net + self.output_bias

    @classmethod
    def init(
        cls,
        input_net: Tensor,
        hidden_net: Tensor,
        hidden_bias: Tensor,
        output_net: Tensor,
        output_bias: Tensor,
        context_size: int,
    ) -> Self:
        return cls(
            input_net.requires_grad_(),
            hidden_net.requires_grad_(),
            hidden_bias.requires_grad_(),
            output_net.requires_grad_(),
            output_bias.requires_grad_(),
            context_size,
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
            randn(hidden_dims, generator=generator),
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
        return cls.init(
            randn(encoding_size, embedding_dims, generator=generator),
            randn(context_length, hidden_dims, generator=generator) * 0.2,
            randn(hidden_dims, generator=generator) * 0.001,
            randn(hidden_dims, encoding_size, generator=generator) * 0.01,
            randn(encoding_size, generator=generator) * 0.001,
            context_size,
        )
