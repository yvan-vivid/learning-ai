from abc import ABC, abstractmethod
from functools import cached_property

from torch import Tensor
from torch.nn.functional import cross_entropy

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
    def forward(self, xis: Tensor) -> Tensor: ...

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

    def run(self, xis: Tensor, yis: Tensor) -> Tensor:
        return self.loss(self.forward(xis), yis)

    def generate(self, xi: Tensor) -> Token:
        return sample_index_logits(self.forward(xi))
