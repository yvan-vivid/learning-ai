from dataclasses import dataclass
from typing import Optional, Self, override

from torch import Tensor, randn

from karpathy_series.makemore.models.components.component import ComponentRecorder
from karpathy_series.makemore.models.sequential import SequentialNet


@dataclass(frozen=True)
class Linear(SequentialNet):
    wa: Tensor

    @override
    def parameters(self) -> list[Tensor]:
        return [self.wa]

    @override
    def forward(self, xis: Tensor, training: bool = False, record: Optional[ComponentRecorder] = None) -> Tensor:
        return self.wa[xis]

    @classmethod
    def init_random_from_size(cls, in_size: int, out_size: int) -> Self:
        return cls(randn((in_size, out_size), requires_grad=True))
