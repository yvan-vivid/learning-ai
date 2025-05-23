from typing import override

from torch import Tensor, no_grad, ones, sqrt, zeros

from karpathy_series.makemore.components.neuro.component import BaseComponent


class BatchNorm1d(BaseComponent):
    eps: float
    momentum: float
    gamma: Tensor
    beta: Tensor
    mean: Tensor
    variance: Tensor

    def __init__(self, dim: int, eps: float = 1e-5, momentum: float = 0.1, init_scale: float = 1.0) -> None:
        self.eps = eps
        self.momentum = momentum
        self.gamma = (ones(dim) * init_scale).requires_grad_()
        self.beta = zeros(dim).requires_grad_()
        self.mean = zeros(dim)
        self.variance = ones(dim)

    def _update_with_momentum(self, register: Tensor, update: Tensor) -> None:
        register.data *= 1 - self.momentum
        register.data += self.momentum * update

    def _update_statistics(self, mean: Tensor, variance: Tensor) -> None:
        with no_grad():
            self._update_with_momentum(self.mean, mean)
            self._update_with_momentum(self.variance, variance)

    def normalize(self, x: Tensor, mean: Tensor, variance: Tensor) -> Tensor:
        return self.gamma * (x - mean) / sqrt(variance + self.eps) + self.beta

    @override
    def forward(self, x: Tensor, training: bool = False) -> Tensor:
        if training:
            self._update_statistics(mean := x.mean(0), variance := x.var(0))
            return self.normalize(x, mean, variance)
        else:
            return self.normalize(x, self.mean, self.variance)

    @override
    def parameters(self) -> list[Tensor]:
        return [self.gamma, self.beta]

    @override
    def describe(self) -> str:
        return f"BatchNorm1d [{self.gamma.shape[0]}]"
