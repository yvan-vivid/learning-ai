from dataclasses import dataclass
from typing import Iterable, List, Tuple, TypeAlias

from torch import Tensor, tensor

from ..bigrams import gen_bigrams
from ..encoding.character import Token
from ..models.net import Net

TrainingSequence: TypeAlias = Iterable[Tuple[Tensor, Tensor]]


def make_training(words: Iterable[List[Token]]) -> Tuple[Tensor, Tensor]:
    xs, ys = [], []
    for word in words:
        for x, y in gen_bigrams(word):
            xs.append(x)
            ys.append(y)
    return (tensor(xs), tensor(ys))


@dataclass(frozen=True)
class Learner:
    net: Net
    lr: float

    def __call__(self, training: TrainingSequence, epochs: int = 1, report_epochs: int = 10) -> None:
        for k in range(epochs):
            for n, ps in enumerate(training):
                loss = self.net.run(*ps)
                self.net.backward(loss)
                self.net.update(self.lr)
            if k % (report_epochs + 1) == 0:
                print(f"Epoch {k} is finished with loss = {loss}")
