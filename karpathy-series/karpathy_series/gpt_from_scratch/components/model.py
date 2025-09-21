from abc import ABC, abstractmethod
from typing import override

from torch import Tensor, concat, multinomial, softmax
from torch.nn import Module


class Model(Module, ABC):
    @override
    @abstractmethod
    def forward(self, ix: Tensor) -> Tensor: ...

    @abstractmethod
    def loss(self, logits: Tensor, targets: Tensor) -> Tensor: ...

    def forward_with_loss(self, ix: Tensor, targets: Tensor) -> tuple[Tensor, Tensor]:
        logits = self.forward(ix)
        return logits, self.loss(logits, targets)

    def generate(self, ix: Tensor, max_output_tokens: int) -> Tensor:
        for _ in range(max_output_tokens):
            logits = self.forward(ix)
            distro = softmax(logits, dim=-1)
            ix_next = multinomial(distro[:, -1, :], 1, replacement=True)
            ix = concat((ix, ix_next), dim=1)
        return ix
