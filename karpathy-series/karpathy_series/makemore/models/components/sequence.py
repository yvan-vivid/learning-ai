from dataclasses import dataclass
from itertools import chain
from typing import Iterable, Optional, override

from torch import Tensor

from karpathy_series.makemore.models.components.component import Component, ComponentRecorder


@dataclass(frozen=True)
class Sequence(Component):
    layers: list[Component]

    @override
    def forward(self, x: Tensor, training: bool = False) -> Tensor:
        return NotImplemented

    @override
    def __call__(self, x: Tensor, training: bool = False, record: Optional[ComponentRecorder] = None) -> Tensor:
        out = x
        for layer in self.layers:
            out = layer(out, training=training, record=record)
        return out

    @override
    def parameters(self) -> Iterable[Tensor]:
        return chain.from_iterable(layer.parameters() for layer in self.layers)

    @override
    def describe(self) -> str:
        descriptions = ", ".join(layer.describe() for layer in self.layers)
        return f"Sequence [{descriptions}]"
