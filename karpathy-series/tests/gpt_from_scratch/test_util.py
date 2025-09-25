from torch import ones, stack, tensor, zeros
from torch.testing import assert_close

from karpathy_series.gpt_from_scratch.util import masked_window

ma = stack(
    [
        tensor([1, 0, 0, 0]),
        tensor([1, 1, 0, 0]) / 2,
        tensor([1, 1, 1, 0]) / 3,
        tensor([1, 1, 1, 1]) / 4,
    ]
)


def test_masked_window() -> None:
    assert_close(masked_window(zeros(4, 4)), ma)
    assert_close(masked_window(ones(4, 4)), ma)


def test_masked_window_batched() -> None:
    assert_close(masked_window(stack([ones(4, 4), zeros(4, 4)])), stack([ma, ma]))
