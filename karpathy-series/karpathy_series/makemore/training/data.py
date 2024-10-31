from dataclasses import dataclass
from random import sample
from typing import Callable, Generic, Iterable, Iterator, Sequence, Tuple, TypeAlias, TypeVar

from torch import Tensor, tensor

from ..encoding.abstract import Encoder
from ..encoding.character import Token
from ..util import block_sequence

IV = TypeVar("IV")
OV = TypeVar("OV")
IT = TypeVar("IT", bound=Token | list[Token])

TrainingSet: TypeAlias = Tuple[Tensor, Tensor]
TrainingSequence: TypeAlias = Iterable[TrainingSet]
TrainingItemGenerator: TypeAlias = Callable[[str], Iterable[Tuple[IV, OV]]]

FreqTrainingSet: TypeAlias = Iterator[Tuple[Token, Token]]


@dataclass(frozen=True)
class TrainingSequencer(Generic[IV, OV, IT]):
    in_encoder: Encoder[IT, IV]
    out_encoder: Encoder[Token, OV]
    generator: TrainingItemGenerator[IV, OV]

    def training_set(self, words: Iterable[str]) -> TrainingSet:
        xs: list[IT] = []
        ys: list[Token] = []
        for word in words:
            for x, y in self.generator(word):
                xs.append(self.in_encoder.encode_or_raise(x))
                ys.append(self.out_encoder.encode_or_raise(y))
        return tensor(xs), tensor(ys)

    def training_sequence(self, words: Sequence[str], block_size: int, shuffle: bool = False) -> TrainingSequence:
        if shuffle:
            words = sample(words, len(words))
        for s in block_sequence(len(words), block_size):
            yield self.training_set(words[s])


@dataclass(frozen=True)
class FreqTrainingSequencer(Generic[IV, OV]):
    in_encoder: Encoder[Token, IV]
    out_encoder: Encoder[Token, OV]
    generator: TrainingItemGenerator[IV, OV]

    def training_sequence(self, words: Iterable[str]) -> FreqTrainingSet:
        for word in words:
            for x, y in self.generator(word):
                if (ix := self.in_encoder.encode(x)) is None:
                    raise ValueError(f"Cannot decode x = {x}")
                if (iy := self.out_encoder.encode(y)) is None:
                    raise ValueError(f"Cannot decode y = {y}")
                yield ix, iy
