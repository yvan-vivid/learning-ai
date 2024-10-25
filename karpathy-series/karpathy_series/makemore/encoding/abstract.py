from abc import ABC, abstractmethod
from typing import Generic, Optional, TypeVar

L = TypeVar("L")
T = TypeVar("T")


class Encoder(ABC, Generic[T, L]):
    size: int

    @abstractmethod
    def encode(self, letter: L) -> Optional[T]: ...

    @abstractmethod
    def decode(self, t: T) -> Optional[L]: ...

    def encode_or_raise(self, letter: L) -> T:
        if (v := self.encode(letter)) is None:
            raise ValueError("encoding failures")
        return v

    def decode_or_raise(self, t: T) -> L:
        if (v := self.decode(t)) is None:
            raise ValueError("decoding failures")
        return v
