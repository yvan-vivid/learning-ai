from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, override

from torch import Tensor, randint, stack


class BatchGenerator(ABC):
    @abstractmethod
    def __call__(self) -> tuple[Tensor, Tensor]: ...

    def reset(self) -> None:
        pass


type BatchFactory = Callable[[Tensor], BatchGenerator]


@dataclass(frozen=True)
class SequentialBlockBatchGenerator(BatchGenerator):
    data: Tensor
    block_size: int
    batch_size: int

    @override
    def __call__(self) -> tuple[Tensor, Tensor]:
        ix = randint(len(self.data) - self.block_size, (self.batch_size,))
        x = stack([self.data[i : i + self.block_size] for i in ix])
        y = stack([self.data[i + 1 : i + self.block_size + 1] for i in ix])
        return (x, y)
