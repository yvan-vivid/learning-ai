from typing import override

from torch import Tensor
from torch.nn import Embedding
from torch.nn.functional import cross_entropy

from karpathy_series.gpt_from_scratch.components.model import Model

# ## Bigram Model with Pytorch Modules
# This is the bigram model involving the construction of a distribution based on the current character.
#
#    P(x[i+1] | x[i]) = softmax(W[x[i+1], x[i]])
#
# This is implemented as an Embedding from tokens into a linear space of dimension |tokens|.


class BigramModel(Model):
    code_size: int
    embedding: Embedding

    def __init__(self, code_size: int) -> None:
        super().__init__()  # type: ignore[reportGeneralTypeIssues]
        self.code_size = code_size
        self.embedding = Embedding(code_size, code_size)

    @override
    def forward(self, ix: Tensor) -> Tensor:
        return self.embedding.forward(ix)

    @override
    def loss(self, logits: Tensor, targets: Tensor) -> Tensor:
        return cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1))
