from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import cached_property
from typing import Iterator, Optional

from torch import Tensor
from torch.nn.functional import cross_entropy

from karpathy_series.makemore.models.components.component import Component

from ..encoding.character import Token
from ..util import sample_index_logits


@dataclass(frozen=True)
class CalcRecorder:
    records: list[tuple[Component, Tensor]] = field(default_factory=list)

    def record(self, component: Component, result: Tensor) -> None:
        self.records.append((component, result))

    def items(self) -> Iterator[tuple[Component, Tensor]]:
        yield from self.records


class SequentialNet(ABC):
    context_size: int

    @abstractmethod
    def parameters(self) -> list[Tensor]: ...

    @cached_property
    def count(self) -> int:
        return sum(p.nelement() for p in self.parameters())

    @abstractmethod
    def forward(self, xis: Tensor, training: bool = False, record: Optional[CalcRecorder] = None) -> Tensor: ...

    def backward(self, loss: Tensor) -> None:
        for wa in self.parameters():
            wa.grad = None
        loss.backward()  # type: ignore[no-untyped-call]

    def update(self, lr: float) -> None:
        for wa in self.parameters():
            if wa.grad is not None:
                wa.data += -lr * wa.grad

    def loss(self, u: Tensor, yis: Tensor) -> Tensor:
        return cross_entropy(u, yis)

    def run(self, xis: Tensor, yis: Tensor, training: bool = True, record: Optional[CalcRecorder] = None) -> Tensor:
        return self.loss(self.forward(xis, training=training, record=record), yis)

    def step(self, xis: Tensor, yis: Tensor, record: Optional[CalcRecorder] = None) -> Tensor:
        self.backward(loss := self.run(xis, yis, training=True, record=record))
        return loss

    def generate(self, xi: Tensor) -> Token:
        return sample_index_logits(self.forward(xi))
