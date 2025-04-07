from dataclasses import dataclass, field
from typing import Self

from torch import Tensor, tensor

from karpathy_series.makemore.components.models.frequentist import FreqModel
from karpathy_series.makemore.components.models.model import Model
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
    model: Model
    lr: float

    def __call__(self, training: TrainingSequence, epochs: int = 1, report_epochs: int = 10) -> LearningRecord:
        record = LearningRecord()
        for k in range(epochs):
            loss = tensor(())
            for _n, (xis, yis) in enumerate(training()):
                loss = self.model.step(xis, yis)
                update_ratios = self.model.update(self.lr)
                record.record(loss, update_ratios)
            if (k + 1) % report_epochs == 0:
                print(f"Epoch {k + 1} is finished with loss = {float(loss.item()): 0.4f}")
        return record


@dataclass(frozen=True)
class FreqLearner:
    model: FreqModel

    def __call__(self, training: TrainingSequence) -> list[float]:
        losses: list[float] = []
        for _n, (xis, yis) in enumerate(training()):
            self.model.train(xis, yis)
            losses.append(float(self.model.run(xis, yis).item()))
        return losses
