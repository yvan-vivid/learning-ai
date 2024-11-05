from torch import Generator, float32, tensor
from torch.testing import assert_close

from karpathy_series.makemore.models.components.linear import Linear

GENERATOR = Generator().manual_seed(123)

TEST_IN = tensor([[1, 2, 3], [3, 4, 5], [5, 6, 7], [9, 10, 11]], dtype=float32)
TEST_OUT = tensor([[6, 8], [10, 12], [14, 16], [22, 24]], dtype=float32)


def test_linear_works() -> None:
    net = Linear(3, 2, generator=GENERATOR)
    net.weight.data = tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=float32)
    assert net.bias is not None
    net.bias.data = tensor([2.0, 3.0], dtype=float32)
    assert_close(net(TEST_IN), TEST_OUT)
