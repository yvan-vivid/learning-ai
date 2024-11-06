from dataclasses import dataclass

from ..models.frequentist import FreqModel
from ..models.sequential import SequentialNet
from .data import TrainingSequence


@dataclass(frozen=True)
class Learner:
    model: SequentialNet
    lr: float

    def __call__(self, training: TrainingSequence, epochs: int = 1, report_epochs: int = 10) -> list[float]:
        losses: list[float] = []
        for k in range(epochs):
            f_loss = 0.0
            for _n, (xis, yis) in enumerate(training()):
                loss = self.model.step(xis, yis)
                self.model.update(self.lr)
                losses.append(f_loss := float(loss.item()))
            if (k + 1) % report_epochs == 0:
                print(f"Epoch {k + 1} is finished with loss = {f_loss}")
        return losses


@dataclass(frozen=True)
class FreqLearner:
    model: FreqModel

    def __call__(self, training: TrainingSequence) -> list[float]:
        losses: list[float] = []
        for _n, (xis, yis) in enumerate(training()):
            self.model.train(xis, yis)
            losses.append(float(self.model.run(xis, yis).item()))
        return losses
