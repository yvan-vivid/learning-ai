from abc import ABC, abstractmethod
from dataclasses import dataclass

from torch import Tensor, arange, tensor

from ..encoding.character import Token
from ..util import sample_index_model
from .embedding import OneHotEnbedding
from .generation import SequenceGenerator
from .net import Net


class SequentialNet(ABC):
    @abstractmethod
    def forward(self, xis: Tensor) -> Tensor: ...

    @abstractmethod
    def backward(self, loss: Tensor) -> None: ...

    @abstractmethod
    def update(self, lr: float) -> None: ...

    def loss(self, u: Tensor, yis: Tensor) -> Tensor:
        """[N x P] x [index(P)] -> []"""
        return -u[arange(u.shape[0]), yis].log().mean()

    def run(self, xis: Tensor, yis: Tensor) -> Tensor:
        return self.loss(self.forward(xis), yis)


@dataclass(frozen=True)
class OneHotNet(SequentialNet, SequenceGenerator):
    embedding: OneHotEnbedding
    net: Net

    def forward(self, xis: Tensor) -> Tensor:
        return self.net.forward(self.embedding(xis))

    def backward(self, loss: Tensor) -> None:
        self.net.backward(loss)

    def update(self, lr: float) -> None:
        self.net.update(lr)

    def generate(self, xi: Token) -> Token:
        return sample_index_model(self.forward(tensor([xi])))
