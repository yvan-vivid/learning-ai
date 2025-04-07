from abc import ABC
from typing import override

from torch import Tensor, no_grad, tensor
from torch.nn.functional import cross_entropy

from karpathy_series.makemore.components.neuro.component import Component, ComponentRecording
from karpathy_series.makemore.components.neuro.sequence import Sequence
from karpathy_series.makemore.encoding.character import Token
from karpathy_series.makemore.util import sample_index_logits


class Model(Component, ABC):
    context_size: int

    def backward(self, loss: Tensor) -> None:
        for wa in self.parameters():
            wa.grad = None
        loss.backward()  # type: ignore[no-untyped-call]

    def update(self, lr: float) -> Tensor:
        data_update_ratios: list[float] = []
        for wa in self.parameters():
            if wa.grad is not None:
                update = lr * wa.grad
                wa.data += -update
                with no_grad():
                    data_update_ratios.append((update.std() / wa.data.std()).item())
        return tensor(data_update_ratios)

    def loss(self, u: Tensor, y: Tensor) -> Tensor:
        return cross_entropy(u, y)

    def run(self, x: Tensor, y: Tensor, training: bool = True, record: ComponentRecording = None) -> Tensor:
        return self.loss(self(x, training=training, record=record), y)

    def step(self, x: Tensor, y: Tensor, record: ComponentRecording = None) -> Tensor:
        self.backward(loss := self.run(x, y, training=True, record=record))
        return loss

    def generate(self, x: Tensor) -> Token:
        return sample_index_logits(self(x))


class SequentialModel(Model, ABC):
    layers: Sequence

    @override
    def parameters(self) -> list[Tensor]:
        return list(self.layers.parameters())

    @override
    def __call__(self, x: Tensor, training: bool = False, record: ComponentRecording = None) -> Tensor:
        return self.layers(x, training, record)
