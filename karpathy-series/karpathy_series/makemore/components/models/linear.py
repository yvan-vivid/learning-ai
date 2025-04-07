from dataclasses import dataclass
from typing import Self, override

from torch import Tensor, randn

from karpathy_series.makemore.components.models.model import Model
from karpathy_series.makemore.components.neuro.component import ComponentRecording


@dataclass(frozen=True)
class Linear(Model):
    wa: Tensor

    @override
    def describe(self) -> str:
        return "A simple linear model without activation"

    @override
    def parameters(self) -> list[Tensor]:
        return [self.wa]

    @override
    def __call__(self, x: Tensor, training: bool = False, record: ComponentRecording = None) -> Tensor:
        return self.wa[x]

    @classmethod
    def init_random_from_size(cls, in_size: int, out_size: int) -> Self:
        return cls(randn((in_size, out_size), requires_grad=True))
