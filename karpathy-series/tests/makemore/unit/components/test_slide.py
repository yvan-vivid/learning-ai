from torch import float32, tensor
from torch.testing import assert_close

from karpathy_series.makemore.components.neuro.slide import Slide
from karpathy_series.makemore.components.typing import Float32

# (2, 4, 2)
TEST_IN = tensor([[[1, 2], [3, 4], [5, 6], [7, 8]], [[7, 8], [9, 10], [11, 12], [13, 14]]], dtype=float32)

# (2, 2, 4)
TEST_OUT = tensor([[[1, 2, 3, 4], [5, 6, 7, 8]], [[7, 8, 9, 10], [11, 12, 13, 14]]], dtype=float32)


def test_slide_low_dim() -> None:
    net = Slide(1, 2)
    in_v = tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    out_v = tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    assert_close(net(in_v), out_v)


def test_slide() -> None:
    net = Slide(1, 2)
    assert_close(net(TEST_IN), TEST_OUT)


def test_slide_shape() -> None:
    net = Slide(1, 2)
    assert net.type_transform(Float32(6, 4)) == Float32(6, 2, 2)
    assert net.type_transform(Float32(2, 6, 4)) == Float32(2, 3, 8)


def test_slide_negative() -> None:
    net = Slide(-2, 2)
    assert_close(net(TEST_IN), TEST_OUT)


def test_slide_negative_shape() -> None:
    net = Slide(-2, 2)
    assert net.type_transform(Float32(6, 4)) == Float32(3, 8)
    assert net.type_transform(Float32(2, 6, 4)) == Float32(2, 3, 8)
