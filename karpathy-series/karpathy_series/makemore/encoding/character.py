from dataclasses import dataclass
from functools import cached_property
from itertools import product
from typing import FrozenSet, Iterable, Optional, Self, override

from ..util import traverse_list, traverse_str
from .abstract import Encoder, TabularEncoder

type Token = int


@dataclass(frozen=True)
class CharacterSet:
    under: FrozenSet[str]
    pad: str

    @cached_property
    def complete(self) -> list[str]:
        return [self.pad] + sorted(self.under)

    @classmethod
    def from_words(cls, words: Iterable[str], pad: str = ".") -> Self:
        return cls(frozenset("".join(words)), pad)


@dataclass(frozen=True)
class CharacterEncoder(TabularEncoder[Token, str]):
    @classmethod
    def from_charset(cls, character_set: CharacterSet) -> Self:
        return cls.from_pairs(enumerate(character_set.complete))


@dataclass(frozen=True)
class BiCharacterEncoder(TabularEncoder[Token, tuple[str, str]]):
    @classmethod
    def from_charset(cls, character_set: CharacterSet) -> Self:
        return cls.from_pairs(enumerate(product(character_set.complete, character_set.complete)))


@dataclass(frozen=True)
class StringEncoder(Encoder[list[Token], str]):
    item_encoder: CharacterEncoder

    @override
    def encode(self, letter: str) -> Optional[list[Token]]:
        return traverse_list(map(self.item_encoder.encode, letter))

    @override
    def decode(self, t: list[Token]) -> Optional[str]:
        return traverse_str(map(self.item_encoder.decode, t))
