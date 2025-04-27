from abc import ABC
from dataclasses import dataclass

from torch import Tensor

from karpathy_series.makemore.components.function.loss import Loss
from karpathy_series.makemore.components.networks.frequentist import Frequentist
from karpathy_series.makemore.components.neuro.component import Component, ComponentRecording


@dataclass(frozen=True)
class Model(ABC):
    component: Component
    loss: Loss

    def __call__(self, x: Tensor, y: Tensor, training: bool = True, record: ComponentRecording = None) -> Tensor:
        """N[s] x M[s] => L"""
        return self.loss(self.component(x, training=training, record=record), y)


@dataclass(frozen=True)
class FreqModel(Model):
    component: Frequentist
