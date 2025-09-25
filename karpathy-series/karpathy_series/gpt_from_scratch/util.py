from torch import Tensor, ones, tril
from torch.nn.functional import softmax


def masked_window(w: Tensor) -> Tensor:
    window_size, final = w.shape[-2], w.shape[-1]
    assert window_size == final

    mask = tril(ones(window_size, window_size))
    return softmax(w.masked_fill(mask == 0, float("-inf")), dim=-1)
