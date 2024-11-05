from torch import float32, tensor
from torch.testing import assert_close

from karpathy_series.makemore.models.components.batch_norm import BatchNorm1d

TEST_IN = tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=float32)
TEST_OUT_TRAIN = tensor([[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]], dtype=float32)
TEST_IN_SINGLE = tensor([1.0, 2.0], dtype=float32)
TEST_OUT_SINGLE = tensor([-1.0, -1.0], dtype=float32)


def test_batch_inference_works() -> None:
    net = BatchNorm1d(2, eps=0.0)
    assert_close(net(TEST_IN), TEST_IN)
    assert_close(net(TEST_IN_SINGLE), TEST_IN_SINGLE)

    net.mean.data = tensor([3.0, 4.0], dtype=float32)
    net.variance.data = tensor([4.0, 4.0], dtype=float32)
    assert_close(net(TEST_IN), TEST_OUT_TRAIN)
    assert_close(net(TEST_IN_SINGLE), TEST_OUT_SINGLE)


def test_batch_training_works() -> None:
    net = BatchNorm1d(2, eps=0.0, momentum=0.5)
    assert_close(net(TEST_IN, training=True), TEST_OUT_TRAIN)
    assert_close(net.mean, tensor([1.5, 2.0], dtype=float32))
    assert_close(net.variance, tensor([2.5, 2.5], dtype=float32))
