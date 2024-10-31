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
    def init_random_from_size(
        cls,
        encoding_size: int,
        context_size: int,
        embedding_dims: int,
        hidden_dims: int,
        generator: Optional[Generator],
    ) -> Self:
        context_length = context_size * embedding_dims
        return cls(
            randn(encoding_size, embedding_dims, generator=generator, requires_grad=True),
            randn(context_length, hidden_dims, generator=generator, requires_grad=True),
            randn(hidden_dims, generator=generator, requires_grad=True),
            randn(hidden_dims, encoding_size, generator=generator, requires_grad=True),
            randn(encoding_size, generator=generator, requires_grad=True),
            context_size,
        )
