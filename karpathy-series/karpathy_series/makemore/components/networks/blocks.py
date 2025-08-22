from collections.abc import Iterable
from dataclasses import dataclass
from typing import Optional, Self, override

from torch import Generator

from karpathy_series.makemore.components.neuro.batch_norm import BatchNorm1d
from karpathy_series.makemore.components.neuro.component import Component, LogitGenerableComponent
from karpathy_series.makemore.components.neuro.functional import Tanh
from karpathy_series.makemore.components.neuro.linear import Linear
from karpathy_series.makemore.components.neuro.sequence import Sequence
from karpathy_series.makemore.util import sliding_window


@dataclass(frozen=True)
class LinearActivationBlock(Sequence):
    fan_in: int
    fan_out: int

    @override
    def describe(self) -> str:
        parts = super().describe()
        return f"A linear activation block: {parts}"

    @classmethod
    def init(
        cls,
        fan_in: int,
        fan_out: int,
        init_scale: float = 1.0,
        normalize: bool = True,
        generator: Optional[Generator] = None,
    ) -> Self:
        layers: list[Component] = [
            Linear(fan_in, fan_out, init_scale=init_scale, generator=generator),
            *((BatchNorm1d(fan_out),) if normalize else ()),
            Tanh(),
        ]
        return cls(layers, fan_in, fan_out)


@dataclass(frozen=True)
class LinearActivationBlocks(Sequence):
    structure: tuple[int, ...]

    @override
    def describe(self) -> str:
        parts = super().describe()
        return f"A linear activation blocks: {parts}"

    @classmethod
    def init(
        cls,
        structure: Iterable[int],
        init_scale: float = 1.0,
        normalize: bool = True,
        generator: Optional[Generator] = None,
    ) -> Self:
        struct = tuple(structure)
        layers: list[Component] = [
            LinearOutputBlock.init(fan_in, fan_out, init_scale=init_scale, normalize=normalize, generator=generator)
            for fan_in, fan_out in sliding_window(struct, 2)
        ]
        return cls(layers, struct)


@dataclass(frozen=True)
class LinearOutputBlock(Sequence, LogitGenerableComponent):
    fan_in: int
    encoding_size: int

    @override
    def describe(self) -> str:
        parts = super().describe()
        return f"A linear output block: {parts}"

    @classmethod
    def init(
        cls,
        fan_in: int,
        encoding_size: int,
        init_scale: float,
        normalize: bool = True,
        generator: Optional[Generator] = None,
    ) -> Self:
        layers: list[Component] = [
            Linear(fan_in, encoding_size, init_scale=init_scale, generator=generator),
            *((BatchNorm1d(encoding_size),) if normalize else ()),
        ]
        return cls(layers, fan_in, encoding_size)
