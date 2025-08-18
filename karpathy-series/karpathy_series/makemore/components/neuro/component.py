from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import override

from torch import Tensor

from karpathy_series.makemore.components.recorder import Recorder
from karpathy_series.makemore.encoding.character import Token
from karpathy_series.makemore.util import sample_index_logits

type ComponentRecorder = Recorder[Component]
type ComponentRecording = ComponentRecorder | None


class Component(ABC):
    @abstractmethod
    def __call__(self, x: Tensor, training: bool = False, record: ComponentRecording = None) -> Tensor:
        """Run component with `training` flag and optional `record` recorder."""
        ...

    @abstractmethod
    def parameters(self) -> Iterable[Tensor]:
        """Return iterable of parameter tensors."""
        ...

    @abstractmethod
    def describe(self) -> str:
        """Give a description of the component."""
        ...

    def size(self) -> int:
        """Number of individual scalar parameters in the component."""
        return sum(p.nelement() for p in self.parameters())

    @abstractmethod
    def shape(self, x: tuple[int, ...]) -> tuple[int, ...]:
        """Shape change."""
        ...


class GenerableComponent(Component, ABC):
    @abstractmethod
    def realize(self, u: Tensor) -> Token: ...

    def generate(self, x: Tensor) -> Token:
        return self.realize(self(x))


class LogitGenerableComponent(GenerableComponent, ABC):
    @override
    def realize(self, u: Tensor) -> Token:
        return sample_index_logits(u)


class BaseComponent(Component, ABC):
    @abstractmethod
    def forward(self, x: Tensor, training: bool = False) -> Tensor:
        """Run forward pass on `x` with `training=True` to run in training mode."""
        ...

    @override
    def __call__(self, x: Tensor, training: bool = False, record: ComponentRecording = None) -> Tensor:
        """Run `forward` and record output if passed a recorder."""
        out = self.forward(x, training)
        if record is not None:
            record(self, out)
            out.retain_grad()
        return out
