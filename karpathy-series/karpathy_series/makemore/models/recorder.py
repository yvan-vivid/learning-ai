from dataclasses import dataclass, field
from typing import Generic, Iterator, TypeVar

from torch import Tensor

C = TypeVar("C")


@dataclass(frozen=True)
class Recorder(Generic[C]):
    records: list[tuple[C, Tensor]] = field(default_factory=list)

    def record(self, rep: C, result: Tensor) -> None:
        self.records.append((rep, result))

    def items(self) -> Iterator[tuple[C, Tensor]]:
        yield from self.records
