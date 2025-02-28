from dataclasses import dataclass, field
from typing import Iterator

from torch import Tensor


@dataclass(frozen=True)
class Recorder[C]:
    records: list[tuple[C, Tensor]] = field(default_factory=list)

    def record(self, rep: C, result: Tensor) -> None:
        self.records.append((rep, result))

    def items(self) -> Iterator[tuple[C, Tensor]]:
        yield from self.records
