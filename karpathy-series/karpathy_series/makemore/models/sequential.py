from abc import ABC, abstractmethod
from functools import cached_property
from typing import Optional

from torch import Tensor, no_grad, tensor
from torch.nn.functional import cross_entropy

from karpathy_series.makemore.models.components.component import ComponentRecorder

from ..encoding.character import Token
from ..util import sample_index_logits


class SequentialNet(ABC):
    context_size: int

    @abstractmethod
    def parameters(self) -> list[Tensor]: ...

    @cached_property
    def count(self) -> int:
        return sum(p.nelement() for p in self.parameters())

    @abstractmethod
    def forward(self, xis: Tensor, training: bool = False, record: Optional[ComponentRecorder] = None) -> Tensor: ...

    def backward(self, loss: Tensor) -> None:
        for wa in self.parameters():
            wa.grad = None
        loss.backward()  # type: ignore[no-untyped-call]

    def update(self, lr: float) -> Tensor:
        data_update_ratios: list[float] = []
        for wa in self.parameters():
            if wa.grad is not None:
                update = lr * wa.grad
                wa.data += -update
                with no_grad():
                    data_update_ratios.append((update.std() / wa.data.std()).item())
        return tensor(data_update_ratios)

    def loss(self, u: Tensor, yis: Tensor) -> Tensor:
        return cross_entropy(u, yis)

    def run(
        self, xis: Tensor, yis: Tensor, training: bool = True, record: Optional[ComponentRecorder] = None
    ) -> Tensor:
        return self.loss(self.forward(xis, training=training, record=record), yis)

    def step(self, xis: Tensor, yis: Tensor, record: Optional[ComponentRecorder] = None) -> Tensor:
        self.backward(loss := self.run(xis, yis, training=True, record=record))
        return loss

    def generate(self, xi: Tensor) -> Token:
        return sample_index_logits(self.forward(xi))
