from typing import override

from torch import Tensor, no_grad, ones, sqrt, zeros

from karpathy_series.makemore.components.neuro.component import BaseComponent


class BatchNorm1d(BaseComponent):
    fan: int
    eps: float
    momentum: float
    gamma: Tensor
    beta: Tensor
    mean: Tensor
    variance: Tensor

    def __init__(self, fan: int, eps: float = 1e-5, momentum: float = 0.1, init_scale: float = 1.0) -> None:
        self.fan = fan
        self.eps = eps
        self.momentum = momentum
        self.gamma = (ones(fan) * init_scale).requires_grad_()
        self.beta = zeros(fan).requires_grad_()
        self.mean = zeros(fan)
        self.variance = ones(fan)

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
            x_f = x.flatten(end_dim=-2)
            self._update_statistics(mean := x_f.mean(0), variance := x_f.var(0))
        else:
            mean, variance = self.mean, self.variance
        return self.normalize(x, mean, variance)

    @override
    def parameters(self) -> list[Tensor]:
        return [self.gamma, self.beta]

    @override
    def describe(self) -> str:
        return f"BatchNorm1d [{self.fan}]"

    @override
    def shape(self, x: tuple[int, ...]) -> tuple[int, ...]:
        assert x[-1] == self.fan, f"last index {x[-1]} not compatible with {self.fan}"
        return x
