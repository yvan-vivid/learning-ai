from dataclasses import dataclass
from functools import reduce
from operator import add
from typing import FrozenSet, Iterable, List, Optional, Self, TypeAlias, TypeVar

from ..util import traverse_list, traverse_str
from .abstract import Encoder
from .character import CharacterEncoder

BOUND_STR = "."

Token: TypeAlias = int
TokenList: TypeAlias = List[Token]

T = TypeVar("T")


@dataclass(frozen=True)
class WordEncoder(Encoder[TokenList, str]):
    letter_encoder: Encoder[Token, str]

    def encode(self, ls: str) -> Optional[TokenList]:
        return traverse_list(map(self.letter_encoder.encode, ls))

    def decode(self, ts: TokenList) -> Optional[str]:
        return traverse_str(map(self.letter_encoder.decode, ts))


@dataclass(frozen=True)
class DelimitedWordEncoder(Encoder[TokenList, str]):
    word_encoder: WordEncoder
    boundary: Token
    token_count: int
    boundary_letter: str = "."

    def encode(self, ls: str) -> Optional[TokenList]:
        if (ts := self.word_encoder.encode(ls)) is None:
            return None
        return [self.boundary] + ts + [self.boundary]

    def decode(self, ts: TokenList) -> Optional[str]:
        if len(ts) < 2 or ts[0] != self.boundary or ts[-1] != self.boundary:
            return None
        return self.word_encoder.decode(ts[1:-1])

    def decode_letter(self, t: Token) -> Optional[str]:
        return self.boundary_letter if t == self.boundary else self.word_encoder.letter_encoder.decode(t)

    def form_token_stream(self, ws: Iterable[TokenList]) -> TokenList:
        return reduce(add, (w + [self.boundary] for w in ws), [self.boundary])

    @classmethod
    def from_charset(cls, tokens: FrozenSet[str]) -> Self:
        ch_encoder = CharacterEncoder.from_charset(tokens)
        delimiter = ch_encoder.size
        return cls(WordEncoder(ch_encoder), delimiter, ch_encoder.size + 1)

    @classmethod
    def from_words(cls, words: Iterable[str]) -> Self:
        return cls.from_charset(frozenset("".join(words)))
