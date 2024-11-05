from torch import float32, tensor
from torch.testing import assert_close

from karpathy_series.makemore.models.components.flatten import Flatten

TEST_IN = tensor([[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]], dtype=float32)
TEST_OUT = tensor([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]], dtype=float32)


def test_flatten() -> None:
    net = Flatten(2)
    assert_close(net(TEST_IN), TEST_OUT)
