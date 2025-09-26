from typing import override

from torch import Tensor
from torch.nn import Embedding

from karpathy_series.gpt_from_scratch.components.network.model import ModelWithCrossEntropyLoss, WindowedGenerable

# ## Bigram Model with Pytorch Modules
# This is the bigram model involving the construction of a distribution based on the current character.
#
#    P(x[i+1] | x[i]) = softmax(W[x[i+1], x[i]])
#
# This is implemented as an Embedding from tokens into a linear space of dimension |tokens|.


class BigramModel(ModelWithCrossEntropyLoss, WindowedGenerable):
    code_size: int
    window_length: int
    embedding: Embedding

    def __init__(self, code_size: int, window_length: int) -> None:
        super().__init__()
        self.code_size = code_size
        self.window_length = window_length
        self.embedding = Embedding(code_size, code_size)

    @override
    def forward(self, ix: Tensor) -> Tensor:
        return self.embedding.forward(ix)
