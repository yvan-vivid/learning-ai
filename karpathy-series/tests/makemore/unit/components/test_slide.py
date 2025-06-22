from torch import float32, tensor
from torch.testing import assert_close

from karpathy_series.makemore.components.neuro.slide import Slide

# (2, 4, 2)
TEST_IN = tensor([[[1, 2], [3, 4], [5, 6], [7, 8]], [[7, 8], [9, 10], [11, 12], [13, 14]]], dtype=float32)

# (2, 2, 4)
TEST_OUT = tensor([[[1, 2, 3, 4], [5, 6, 7, 8]], [[7, 8, 9, 10], [11, 12, 13, 14]]], dtype=float32)


def test_flatten_low_dim() -> None:
    net = Slide(1, 2)
    in_v = tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    out_v = tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    assert_close(net(in_v), out_v)


def test_flatten() -> None:
    net = Slide(1, 2)
    assert_close(net(TEST_IN), TEST_OUT)
