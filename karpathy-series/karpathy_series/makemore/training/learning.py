from dataclasses import dataclass
from typing import Callable

from ..models.frequentist import FreqModel
from ..models.net import Net
from .data import FreqTrainingSet, TrainingSequence


@dataclass(frozen=True)
class Learner:
    model: Net
    lr: float

    def __call__(self, training: Callable[[], TrainingSequence], epochs: int = 1, report_epochs: int = 10) -> None:
        for k in range(epochs):
            for n, ps in enumerate(training()):
                loss = self.model.run(*ps)
                self.model.backward(loss)
                self.model.update(self.lr)
            if (k + 1) % report_epochs == 0:
                print(f"Epoch {k + 1} is finished with loss = {loss}")


@dataclass(frozen=True)
class FreqLearner:
    model: FreqModel

    def __call__(self, training: FreqTrainingSet) -> None:
        for ix, iy in training:
            self.model.under[ix, iy] += 1
