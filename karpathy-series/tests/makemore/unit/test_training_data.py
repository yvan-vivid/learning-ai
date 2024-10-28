from functools import partial

from torch import tensor
from torch.testing import assert_close

from karpathy_series.makemore.bigrams import BiGram
from karpathy_series.makemore.encoding.character import CharacterEncoder, CharacterSet
from karpathy_series.makemore.training.data import TrainingSequencer

CHAR_SET = CharacterSet.from_words(["abc"])
ENC = CharacterEncoder.from_charset(CHAR_SET)
TS = TrainingSequencer(ENC, ENC, partial(BiGram.generate, "."))


def test_make_training_set() -> None:
    in_vec, out_vec = TS.training_set(["a", "bc", "b"])
    assert_close(in_vec, tensor([0, 1, 0, 2, 3, 0, 2]))
    assert_close(out_vec, tensor([1, 0, 2, 3, 0, 2, 0]))


def test_make_training_sequence() -> None:
    outs = tuple(TS.training_sequence(["a", "bc", "b"], 2))
    assert_close(outs[0][0], tensor([0, 1, 0, 2, 3]))
    assert_close(outs[0][1], tensor([1, 0, 2, 3, 0]))
    assert_close(outs[1][0], tensor([0, 2]))
    assert_close(outs[1][1], tensor([2, 0]))
