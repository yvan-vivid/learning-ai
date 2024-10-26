from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Self

from torch import Tensor, randn

from ..util import sliding_window


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
class MultiLayer(Net):
    was: List[Tensor]  # [[M x H1], [H1 x H2], ..., [HN, P]]

    def forward(self, xs: Tensor) -> Tensor:
        """[N x M] -> [N x P]"""
        m = xs
        for wa in self.was:
            m = softmax(m @ wa)
        return m

    def backward(self, loss: Tensor) -> None:
        for wa in self.was:
            wa.grad = None
        loss.backward()  # type: ignore[no-untyped-call]

    def update(self, lr: float) -> None:
        for wa in self.was:
            if wa.grad is not None:
                wa.data += -lr * wa.grad

    @classmethod
    def init_random_from_size(cls, size: int, hidden: List[int]) -> Self:
        hidden.insert(0, size)
        hidden.append(size)
        return cls([randn(p, requires_grad=True) for p in sliding_window(hidden, 2)])
