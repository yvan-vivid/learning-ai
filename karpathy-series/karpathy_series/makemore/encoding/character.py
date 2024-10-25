from dataclasses import dataclass
from typing import Dict, FrozenSet, Optional, Self, TypeAlias

from .abstract import Encoder

Token: TypeAlias = int


@dataclass(frozen=True)
class CharacterEncoder(Encoder[Token, str]):
    forward: Dict[str, Token]
    reverse: Dict[Token, str]
    size: int

    def encode(self, t: str) -> Optional[Token]:
        return self.forward.get(t)

    def decode(self, c: Token) -> Optional[str]:
        return self.reverse.get(c)

    @classmethod
    def from_charset(cls, tokens: FrozenSet[str]) -> Self:
        charlist = sorted(tokens)
        forward = {c: i for i, c in enumerate(charlist)}
        reverse = dict(enumerate(charlist))
        return cls(forward, reverse, len(forward))
