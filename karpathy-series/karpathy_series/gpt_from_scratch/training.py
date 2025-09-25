from dataclasses import dataclass

from torch import Tensor, no_grad, zeros
from torch.optim import Optimizer

from karpathy_series.gpt_from_scratch.batch_generation import BatchFactory, BatchGenerator
from karpathy_series.gpt_from_scratch.components.model import Model
from karpathy_series.gpt_from_scratch.data import SequenceData


@dataclass(frozen=True)
class Stepper:
    model: Model
    optimizer: Optimizer
    batcher: BatchGenerator

    def __call__(self) -> Tensor:
        _, loss = self.model.forward_with_loss(*self.batcher())
        self.optimizer.zero_grad(set_to_none=True)
        _ = loss.backward()  # type: ignore
        self.optimizer.step()
        return loss


@dataclass(frozen=True)
class LossEstimator:
    model: Model
    batching: BatchFactory
    evals: int = 100

    @no_grad()
    def __call__(self, data: Tensor) -> Tensor:
        batcher = self.batching(data)
        _ = self.model.eval()
        losses = zeros(self.evals)
        for k in range(self.evals):
            _, loss = self.model.forward_with_loss(*batcher())
            losses[k] = loss.item()
        _ = self.model.train()
        return losses.mean()

    def report(self, data: SequenceData, step: int) -> None:
        training_loss = self(data.training)
        validation_loss = self(data.validation)
        print(
            ", ".join(
                (
                    f"step = {step}",
                    f"training loss = {training_loss.item():.3f}",
                    f"validation loss = {validation_loss.item():.3f}",
                )
            )
        )


@dataclass(frozen=True)
class Trainer:
    model: Model
    optimizer: Optimizer
    batching: BatchFactory
    data: SequenceData

    def train(self, steps: int, report_steps: int = 10, eval_losses: int = 100) -> None:
        report_freq = steps // report_steps
        stepper = Stepper(self.model, self.optimizer, self.batching(self.data.training))
        estimator = LossEstimator(self.model, self.batching, eval_losses)
        for step in range(steps):
            _ = stepper()
            if step % report_freq == 0:
                estimator.report(self.data, step)
