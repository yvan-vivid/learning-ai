from dataclasses import dataclass, field
from typing import Self

from torch import Tensor, no_grad, tensor

from karpathy_series.makemore.components.models.model import FreqModel, NetModel
from karpathy_series.makemore.training.data import TrainingSequence


@dataclass(frozen=True)
class LearningRecord:
    loss: list[float] = field(default_factory=list)
    update_ratios: list[list[float]] = field(default_factory=list)

    def __add__(self, other: Self) -> Self:
        return self.__class__(self.loss + other.loss, self.update_ratios + other.update_ratios)

    def record_loss(self, loss: Tensor) -> None:
        self.loss.append(float(loss.item()))

    def record_update_ratios(self, update_ratios: Tensor) -> None:
        self.update_ratios.append([float(f) for f in update_ratios.log10()])

    def record(self, loss: Tensor, update_ratios: Tensor) -> None:
        self.record_loss(loss)
        self.record_update_ratios(update_ratios)


@dataclass(frozen=True)
class Learner:
    model: NetModel
    lr: float

    def update(self, lr: float) -> Tensor:
        data_update_ratios: list[float] = []
        for wa in self.model.component.parameters():
            if wa.grad is not None:
                update = lr * wa.grad
                wa.data += -update
                with no_grad():
                    data_update_ratios.append((update.std() / wa.data.std()).item())
        return tensor(data_update_ratios)

    def __call__(self, training: TrainingSequence, epochs: int = 1, report_epochs: int = 10) -> LearningRecord:
        record = LearningRecord()
        for k in range(epochs):
            loss = tensor(())
            for _n, (xis, yis) in enumerate(training()):
                loss = self.model.run(xis, yis, training=True)
                self.model.backward(loss)
                update_ratios = self.update(self.lr)
                record.record(loss, update_ratios)
            if (k + 1) % report_epochs == 0:
                print(f"Epoch {k + 1} is finished with loss = {float(loss.item()): 0.4f}")
        return record


@dataclass(frozen=True)
class FreqLearner:
    model: FreqModel

    def train(self, xis: Tensor, yis: Tensor) -> None:
        assert xis.shape == yis.shape
        for xi, yi in zip(list(xis), list(yis)):
            self.model.counts[xi, yi] += 1

    def __call__(self, training: TrainingSequence) -> list[float]:
        losses: list[float] = []
        for _n, (xis, yis) in enumerate(training()):
            loss = self.model.run(xis, yis)
            losses.append(loss.item())
            self.train(xis, yis)
        return losses
