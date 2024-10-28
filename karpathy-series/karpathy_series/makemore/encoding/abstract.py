from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Generic, Iterable, List, Optional, Self, Tuple, TypeVar

from ..util import traverse_list

L = TypeVar("L")
T = TypeVar("T")


class Encoder(ABC, Generic[T, L]):
    size: int

    @abstractmethod
    def encode(self, letter: L) -> Optional[T]: ...

    @abstractmethod
    def decode(self, t: T) -> Optional[L]: ...

    def encodes(self, letters: Iterable[L]) -> Optional[List[T]]:
        return traverse_list(map(self.encode, letters))

    def decodes(self, tokens: Iterable[T]) -> Optional[List[L]]:
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
    forward: Dict[L, T]
    reverse: Dict[T, L]
    size: int

    def encode(self, t: L) -> Optional[T]:
        return self.forward.get(t)

    def decode(self, c: T) -> Optional[L]:
        return self.reverse.get(c)

    @classmethod
    def from_pairs(cls, token_letter_pairs: Iterable[Tuple[T, L]]) -> Self:
        forward, reverse = dict(), dict()
        for token, letter in token_letter_pairs:
            forward[letter] = token
            reverse[token] = letter
        return cls(forward, reverse, len(forward))
