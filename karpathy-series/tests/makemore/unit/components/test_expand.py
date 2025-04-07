from torch import float32, tensor
from torch.testing import assert_close

from karpathy_series.makemore.components.neuro.expand import Expand

TEST_IN = tensor([[[1, 2], [3, 4], [5, 6], [7, 8]], [[7, 8], [9, 10], [11, 12], [13, 14]]], dtype=float32)
TEST_OUT = tensor([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [[[7, 8], [9, 10]], [[11, 12], [13, 14]]]], dtype=float32)


def test_flatten() -> None:
    net = Expand(1, 2)
    assert_close(net(TEST_IN), TEST_OUT)
