from collections.abc import Iterable
from dataclasses import dataclass
from itertools import chain
from typing import override

from torch import Tensor

from karpathy_series.makemore.components.neuro.component import Component, ComponentRecording


@dataclass(frozen=True)
class Sequence(Component):
    layers: list[Component]

    @override
    def __call__(self, x: Tensor, training: bool = False, record: ComponentRecording = None) -> Tensor:
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

    @override
    def shape(self, x: tuple[int, ...]) -> tuple[int, ...]:
        s = x
        for layer in self.layers:
            s = layer.shape(s)
        return s
