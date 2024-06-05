"""Training"""
from dataclasses import dataclass
from typing import Dict, Iterable, List, Self, Set


@dataclass(frozen=True)
class CharacterEncoding:
    """Character encoding"""
    stoi: Dict[str, int]
    itos: Dict[int, str]

    def encode(self, s: str) -> List[int]:
        """encode"""
        return [self.stoi[ch] for ch in s]

    def decode(self, v: Iterable[int]) -> str:
        """decode"""
        return ''.join(self.itos[i] for i in v)

    @classmethod
    def from_character_set(cls, chars: Set[str]) -> Self:
        """Create encoding from set of characters"""
        ordered = sorted(chars)
        return cls({ch: i for i, ch in enumerate(ordered)}, dict(enumerate(ordered)))
