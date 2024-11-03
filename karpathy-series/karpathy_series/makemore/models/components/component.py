from abc import ABC, abstractmethod

from torch import Tensor


class Component(ABC):
    @abstractmethod
    def __call__(self, x: Tensor, training: bool = False) -> Tensor: ...

    @abstractmethod
    def parameters(self) -> list[Tensor]: ...

    @abstractmethod
    def describe(self) -> str: ...

    def size(self) -> int:
        return sum(p.nelement() for p in self.parameters())
