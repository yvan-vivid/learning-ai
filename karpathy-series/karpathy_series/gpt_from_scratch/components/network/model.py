from abc import ABC, abstractmethod
from typing import override

from torch import Tensor, concat, multinomial, softmax, zeros
from torch.nn import Module
from torch.nn.functional import cross_entropy


class Model(Module, ABC):
    @override
    @abstractmethod
    def forward(self, ix: Tensor) -> Tensor: ...

    @abstractmethod
    def loss(self, logits: Tensor, targets: Tensor) -> Tensor: ...

    def forward_with_loss(self, ix: Tensor, targets: Tensor) -> tuple[Tensor, Tensor]:
        logits = self.forward(ix)
        return logits, self.loss(logits, targets)


class Generable(Model, ABC):
    @abstractmethod
    def generate(self, state: Tensor, max_output_tokens: int) -> Tensor: ...


class WindowedGenerable(Generable, ABC):
    window_length: int

    def initial(self) -> Tensor:
        return zeros(1, self.window_length)

    @override
    def generate(self, state: Tensor, max_output_tokens: int) -> Tensor:
        for _ in range(max_output_tokens):
            logits = self.forward(state[:, -self.window_length :])
            distro = softmax(logits, dim=-1)
            ix_next = multinomial(distro[:, -1, :], 1, replacement=True)
            state = concat((state, ix_next), dim=1)
        return state


class ModelWithCrossEntropyLoss(Model, ABC):
    @override
    def loss(self, logits: Tensor, targets: Tensor) -> Tensor:
        return cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1))
