from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterator, override

from torch import Tensor


class Recorder[C](ABC):
    @abstractmethod
    def __call__(self, rep: C, result: Tensor) -> None: ...


@dataclass(frozen=True)
class FreeRecorder[C](Recorder[C]):
    records: list[tuple[C, Tensor]] = field(default_factory=list)

    @override
    def __call__(self, rep: C, result: Tensor) -> None:
        self.records.append((rep, result))

    def items(self) -> Iterator[tuple[C, Tensor]]:
        yield from self.records
