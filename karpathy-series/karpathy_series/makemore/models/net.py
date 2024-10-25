from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Self

from torch import Tensor, randn

from .training import TrainingSequence


def softmax(v: Tensor) -> Tensor:
    p = v.exp()
    return p / p.sum(1, keepdim=True)


class Net(ABC):
    @abstractmethod
    def forward(self, xs: Tensor) -> Tensor: ...

    @abstractmethod
    def backward(self, loss: Tensor) -> None: ...

    @abstractmethod
    def update(self, lr: float) -> None: ...

    def loss(self, u: Tensor, ys: Tensor) -> Tensor:
        """[N x P] x [N x P] -> []"""
        return -(u * ys).sum(dim=1).log().mean()

    def run(self, xs: Tensor, yis: Tensor) -> Tensor:
        return self.loss(self.forward(xs), yis)


@dataclass(frozen=True)
class OneLayer(Net):
    wa: Tensor  # [M x P]

    def forward(self, xs: Tensor) -> Tensor:
        """[N x M] -> [N x P]"""
        return softmax(xs @ self.wa)

    def backward(self, loss: Tensor) -> None:
        self.wa.grad = None
        loss.backward()  # type: ignore[no-untyped-call]

    def update(self, lr: float) -> None:
        if self.wa.grad is not None:
            self.wa.data += -lr * self.wa.grad

    @classmethod
    def init_random_from_size(cls, size: int) -> Self:
        return cls(randn((size, size), requires_grad=True))


@dataclass(frozen=True)
class Learner:
    net: Net
    lr: float

    def __call__(self, training: TrainingSequence, epochs: int = 1) -> None:
        for k in range(epochs):
            print(f"Epoch {k} is running:")
            for n, ps in enumerate(training):
                loss = self.net.run(*ps)
                if n % 20 == 0:
                    print(f"    loss = {loss}")
                self.net.backward(loss)
                self.net.update(self.lr)
            print(f"Epoch {k} is finished with loss = {loss}")