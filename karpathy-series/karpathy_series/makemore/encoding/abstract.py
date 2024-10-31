from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, Iterable, Optional, Self, Tuple, TypeVar, override

from ..util import traverse_list

L = TypeVar("L")
T = TypeVar("T")


class Encoder(ABC, Generic[T, L]):
    size: int

    @abstractmethod
    def encode(self, letter: L) -> Optional[T]: ...

    @abstractmethod
    def decode(self, t: T) -> Optional[L]: ...

    def encodes(self, letters: Iterable[L]) -> Optional[list[T]]:
        return traverse_list(map(self.encode, letters))

    def decodes(self, tokens: Iterable[T]) -> Optional[list[L]]:
        return traverse_list(map(self.decode, tokens))

    def encode_or_raise(self, letter: L) -> T:
        if (v := self.encode(letter)) is None:
            raise ValueError("encoding failures")
        return v

    def decode_or_raise(self, t: T) -> L:
        if (v := self.decode(t)) is None:
            raise ValueError("decoding failures")
        return v


@dataclass(frozen=True)
class TabularEncoder(Generic[T, L], Encoder[T, L]):
    forward: dict[L, T]
    reverse: dict[T, L]
    size: int

    @override
    def encode(self, letter: L) -> Optional[T]:
        return self.forward.get(letter)

    @override
    def decode(self, t: T) -> Optional[L]:
        return self.reverse.get(t)

    @classmethod
    def from_pairs(cls, token_letter_pairs: Iterable[Tuple[T, L]]) -> Self:
        forward: dict[L, T] = dict()
        reverse: dict[T, L] = dict()
        for token, letter in token_letter_pairs:
            forward[letter] = token
            reverse[token] = letter
        return cls(forward, reverse, len(forward))
