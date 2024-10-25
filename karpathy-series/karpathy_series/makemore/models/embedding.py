from abc import ABC, abstractmethod
from dataclasses import dataclass

from torch import Tensor
from torch.nn.functional import one_hot


class Embedding(ABC):
    @abstractmethod
    def __call__(self, tokens: Tensor) -> Tensor: ...


@dataclass(frozen=True)
class OneHotEnbedding(Embedding):
    encoding_size: int

    def __call__(self, tokens: Tensor) -> Tensor:
        """[N] -> [N x M] where M = encoding_size"""
        return one_hot(tokens, num_classes=self.encoding_size).float()
