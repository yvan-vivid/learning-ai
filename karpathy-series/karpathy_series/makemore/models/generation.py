from dataclasses import dataclass
from functools import cached_property, partial
from typing import Callable

from torch import tensor

from karpathy_series.makemore.encoding.abstract import Encoder
from karpathy_series.makemore.encoding.character import CharacterSet, Token
from karpathy_series.makemore.models.sequential import SequentialNet


def generate_string[V](
    feedback: Callable[[V], Callable[[str], V]],
    forward: Callable[[V], str],
    initial: V,
    final: str,
    max_length: int = 100,
) -> str:
    fb_state = feedback(in_v := initial)
    out = ""
    for _k in range(max_length):
        if (out_v := forward(in_v)) == final:
            break
        out += out_v
        in_v = fb_state(out_v)
    return out


def feedbacker[V](update: Callable[[V, str], V], initial: V) -> Callable[[str], V]:
    state = initial

    def _inner(c: str) -> V:
        nonlocal state
        state = update(state, c)
        return state

    return _inner


def _n_gram_update(state: str, c: str) -> str:
    return state[1:] + c


def _tri_gram_update(state: tuple[str, str], c: str) -> tuple[str, str]:
    return state[1], c


n_gram_generate = partial(generate_string, partial(feedbacker, _n_gram_update))
bi_gram_generate = partial(generate_string, partial(feedbacker, (lambda _state, c: c)))
tri_gram_generate = partial(generate_string, partial(feedbacker, _tri_gram_update))


@dataclass(frozen=True)
class BiGramNetGenerator:
    charset: CharacterSet
    encoder: Encoder[Token, str]
    net: SequentialNet

    @cached_property
    def generator(self) -> Callable[[int], str]:
        return partial(bi_gram_generate, self.forward, self.charset.pad, self.charset.pad)

    def forward(self, c: str) -> str:
        in_v = tensor([self.encoder.encode_or_raise(c)])
        return self.encoder.decode_or_raise(self.net.generate(in_v))

    def __call__(self, max_length: int = 100) -> str:
        return self.generator(max_length)


@dataclass(frozen=True)
class TriGramNetGenerator:
    charset: CharacterSet
    in_encoder: Encoder[Token, tuple[str, str]]
    out_encoder: Encoder[Token, str]
    net: SequentialNet

    @cached_property
    def generator(self) -> Callable[[int], str]:
        return partial(tri_gram_generate, self.forward, (self.charset.pad, self.charset.pad), self.charset.pad)

    def forward(self, c: tuple[str, str]) -> str:
        in_v = tensor([self.in_encoder.encode_or_raise(c)])
        return self.out_encoder.decode_or_raise(self.net.generate(in_v))

    def __call__(self, max_length: int = 100) -> str:
        return self.generator(max_length)


@dataclass(frozen=True)
class NGramNetGenerator:
    charset: CharacterSet
    in_encoder: Encoder[list[Token], str]
    out_encoder: Encoder[Token, str]
    net: SequentialNet

    @cached_property
    def generator(self) -> Callable[[int], str]:
        size = self.net.context_size
        return partial(n_gram_generate, self.forward, self.charset.pad * size, self.charset.pad)

    def forward(self, c: str) -> str:
        in_v = tensor(self.in_encoder.encode_or_raise(c))
        return self.out_encoder.decode_or_raise(self.net.generate(in_v))

    def __call__(self, max_length: int = 100) -> str:
        return self.generator(max_length)
