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
class Trainer:
    model: Model
    optimizer: Optimizer
    batching: BatchFactory
    data: SequenceData

    @no_grad()  # type: ignore
    def estimate_loss(self, eval_losses: int) -> Tensor:
        _ = self.model.eval()
        batcher = self.batching(self.data.validation)
        losses = zeros(eval_losses)
        for k in range(eval_losses):
            _, loss = self.model.forward_with_loss(*batcher())
            losses[k] = loss.item()
        _ = self.model.train()
        return losses.mean()

    def train(self, steps: int, report_steps: int = 10, eval_losses: int = 100) -> None:
        batcher = self.batching(self.data.training)
        report_freq = steps // report_steps
        stepper = Stepper(self.model, self.optimizer, batcher)
        for step in range(steps):
            _ = stepper()
            if step % report_freq == 0:
                estimated_loss = self.estimate_loss(eval_losses)
                print(f"step = {step}, validation loss = {estimated_loss.item()}")
