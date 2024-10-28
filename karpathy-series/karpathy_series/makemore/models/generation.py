from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, Optional, Tuple, Type, TypeVar

from ..encoding.abstract import Encoder
from ..encoding.character import Token

IV = TypeVar("IV")
OV = TypeVar("OV")


class SequenceGenerator(ABC):
    @abstractmethod
    def generate(self, xi: Token) -> Token: ...


class GeneratorFeedback(Generic[IV, OV], ABC):
    def __init__(self, xi: IV) -> None:
        pass

    @abstractmethod
    def __call__(self, yi: OV) -> IV: ...


@dataclass(frozen=True)
class StringGenerator(Generic[IV]):
    in_encoder: Encoder[Token, IV]
    out_encoder: Encoder[Token, str]
    feedback: Type[GeneratorFeedback[IV, str]]
    start: IV
    end: str

    def __call__(self, model: SequenceGenerator, max_length: int = 100) -> str:
        def _generate(v: IV) -> Optional[str]:
            if (ix := self.in_encoder.encode(v)) is None:
                return None

            if (w := self.out_encoder.decode(model.generate(ix))) is None:
                return None

            return w

        fb = self.feedback(current := self.start)
        name = ""
        for k in range(max_length):
            out = _generate(current)
            if out == self.end:
                break
            if out is None:
                raise ValueError("Encoding problem")
            current = fb(out)
            name += out
        return name


class BiGramFeedback(GeneratorFeedback[str, str]):
    def __call__(self, yi: str) -> str:
        return yi


class TriGramFeedback(GeneratorFeedback[Tuple[str, str], str]):
    register: str

    def __init__(self, xi: Tuple[str, str]) -> None:
        self.register = xi[1]

    def __call__(self, yi: str) -> Tuple[str, str]:
        out = (self.register, yi)
        self.register = yi
        return out
