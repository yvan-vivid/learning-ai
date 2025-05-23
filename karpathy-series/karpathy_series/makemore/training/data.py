from dataclasses import dataclass
from itertools import batched
from random import sample
from typing import Callable, Iterable, Self, Sequence, Tuple

from torch import Tensor, tensor

from ..encoding.abstract import Encoder
from ..encoding.character import Token

type TrainingSet = Tuple[Tensor, Tensor]
type TrainingSequence = Callable[[], Iterable[TrainingSet]]
type TrainingItemGenerator[IV, OV] = Callable[[str], Iterable[Tuple[IV, OV]]]


@dataclass(frozen=True)
class TrainingSequencer[IV, OV, IT: Token | list[Token]]:
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
        return lambda: map(self.training_set, batched(sample(words, len(words)) if shuffle else words, block_size))


@dataclass(frozen=True)
class DataSplit[IV]:
    training: list[IV]
    validation: list[IV]
    development: list[IV]

    @classmethod
    def split(cls, data: list[IV], train: float, test: float, dev: float) -> Self:
        total = train + test + dev
        train, test, dev = train / total, test / total, dev / total
        data = sample(data, len(data))
        train_size, test_size = int(train * len(data)), int(test * len(data))
        test_end = train_size + test_size
        return cls(data[:train_size], data[train_size:test_end], data[test_end:])
