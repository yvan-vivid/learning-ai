from torch import tensor
from torch.testing import assert_close

from karpathy_series.makemore.models.embedding import OneHotEnbedding

CODE_SIZE = 3
ONE_HOT = OneHotEnbedding(CODE_SIZE)


def test_one_hot_embedding() -> None:
    index = tensor([0, 1, 0, 2, 1])
    encoded = ONE_HOT(index)
    assert encoded.shape == (5, 3)
    assert_close(
        encoded,
        tensor(
            [
                [1, 0, 0],
                [0, 1, 0],
                [1, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
            ]
        ).float(),
    )
