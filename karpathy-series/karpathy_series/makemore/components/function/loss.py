from abc import ABC, abstractmethod
from typing import override

from torch import Tensor
from torch.nn.functional import cross_entropy

from karpathy_series.makemore.util import cross_entropy_exp


class Loss(ABC):
    @abstractmethod
    def __call__(self, logits: Tensor, actual: Tensor) -> Tensor: ...


class CrossEntropyLoss(Loss):
    @override
    def __call__(self, logits: Tensor, actual: Tensor) -> Tensor:
        return cross_entropy(logits, actual)


class CrossEntropyExpLoss(Loss):
    @override
    def __call__(self, logits: Tensor, actual: Tensor) -> Tensor:
        return cross_entropy_exp(logits, actual)
