from abc import ABC, abstractmethod
from typing import Iterable, Optional, override

from torch import Tensor

from karpathy_series.makemore.models.recorder import Recorder

type ComponentRecorder = Recorder[Component]


class Component(ABC):
    @abstractmethod
    def __call__(self, x: Tensor, training: bool = False, record: Optional[ComponentRecorder] = None) -> Tensor:
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


class BaseComponent(Component, ABC):
    @abstractmethod
    def forward(self, x: Tensor, training: bool = False) -> Tensor:
        """Run forward pass on `x` with `training=True` to run in training mode."""
        ...

    @override
    def __call__(self, x: Tensor, training: bool = False, record: Optional[ComponentRecorder] = None) -> Tensor:
        """
        Run `forward` and record output if passed a recorder.
        """
        out = self.forward(x, training)
        if record is not None:
            record.record(self, out)
            out.retain_grad()
        return out
