from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property, partial
from typing import Callable, Optional, Self, override

from torch import Generator, Tensor, arange, randn, tanh, tensor
from torch.nn.functional import cross_entropy

from ..encoding.abstract import Encoder
from ..encoding.character import CharacterSet, Token
from ..util import sample_index_logits, sample_index_model
from .embedding import OneHotEnbedding
from .generation import bi_gram_generate, n_gram_generate, tri_gram_generate
from .net import Net


class SequentialNet(ABC):
    @abstractmethod
    def forward(self, xis: Tensor) -> Tensor: ...

    @abstractmethod
    def backward(self, loss: Tensor) -> None: ...

    @abstractmethod
    def update(self, lr: float) -> None: ...

    def loss(self, u: Tensor, yis: Tensor) -> Tensor:
        """[N x P] x [index(P)] -> []"""
        return -u[arange(u.shape[0]), yis].log().mean()

    def run(self, xis: Tensor, yis: Tensor) -> Tensor:
        return self.loss(self.forward(xis), yis)


@dataclass(frozen=True)
class OneHotNet(SequentialNet):
    embedding: OneHotEnbedding
    net: Net

    @override
    def forward(self, xis: Tensor) -> Tensor:
        return self.net.forward(self.embedding(xis))

    @override
    def backward(self, loss: Tensor) -> None:
        self.net.backward(loss)

    @override
    def update(self, lr: float) -> None:
        self.net.update(lr)

    def generate(self, xi: Tensor) -> Token:
        return sample_index_model(self.forward(xi))


@dataclass(frozen=True)
class MPLNet(SequentialNet):
    input_net: Tensor
    hidden_net: Tensor
    hidden_bias: Tensor
    output_net: Tensor
    output_bias: Tensor
    context_size: int

    @cached_property
    def parameters(self) -> list[Tensor]:
        return [self.input_net, self.hidden_net, self.hidden_bias, self.output_net, self.output_bias]

    @cached_property
    def count(self) -> int:
        return sum(p.nelement() for p in self.parameters)

    @override
    def forward(self, xis: Tensor) -> Tensor:
        context_length = self.hidden_net.shape[0]
        w = self.input_net[xis].view(-1, context_length)
        h = tanh(w @ self.hidden_net + self.hidden_bias)
        return h @ self.output_net + self.output_bias

    @override
    def backward(self, loss: Tensor) -> None:
        for wa in self.parameters:
            wa.grad = None
        loss.backward()  # type: ignore[no-untyped-call]

    @override
    def loss(self, u: Tensor, yis: Tensor) -> Tensor:
        """[N x P] x [index(P)] -> []"""
        return cross_entropy(u, yis)

    @override
    def update(self, lr: float) -> None:
        for wa in self.parameters:
            if wa.grad is not None:
                wa.data += -lr * wa.grad

    def generate(self, xi: Tensor) -> Token:
        return sample_index_logits(self.forward(xi))

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


@dataclass(frozen=True)
class BiGramNetGenerator:
    charset: CharacterSet
    encoder: Encoder[Token, str]
    net: OneHotNet

    @cached_property
    def generator(self) -> Callable[[int], str]:
        return partial(bi_gram_generate, self.forward, self.charset.pad, self.charset.pad)

    def forward(self, c: str) -> str:
        in_v = tensor([self.encoder.encode_or_raise(c)])
        out_v = self.net.forward(in_v)
        return self.encoder.decode_or_raise(sample_index_model(out_v))

    def __call__(self, max_length: int = 100) -> str:
        return self.generator(max_length)


@dataclass(frozen=True)
class TriGramNetGenerator:
    charset: CharacterSet
    in_encoder: Encoder[Token, tuple[str, str]]
    out_encoder: Encoder[Token, str]
    net: OneHotNet

    @cached_property
    def generator(self) -> Callable[[int], str]:
        return partial(tri_gram_generate, self.forward, (self.charset.pad, self.charset.pad), self.charset.pad)

    def forward(self, c: tuple[str, str]) -> str:
        in_v = tensor([self.in_encoder.encode_or_raise(c)])
        out_v = self.net.forward(in_v)
        return self.out_encoder.decode_or_raise(sample_index_model(out_v))

    def __call__(self, max_length: int = 100) -> str:
        return self.generator(max_length)


@dataclass(frozen=True)
class NGramNetGenerator:
    charset: CharacterSet
    in_encoder: Encoder[list[Token], str]
    out_encoder: Encoder[Token, str]
    net: MPLNet

    @cached_property
    def generator(self) -> Callable[[int], str]:
        size = self.net.context_size
        return partial(n_gram_generate, self.forward, self.charset.pad * size, self.charset.pad)

    def forward(self, c: str) -> str:
        in_v = tensor(self.in_encoder.encode_or_raise(c))
        out_v = self.net.forward(in_v)
        return self.out_encoder.decode_or_raise(sample_index_logits(out_v))

    def __call__(self, max_length: int = 100) -> str:
        return self.generator(max_length)
