from karpathy_series.makemore.bigrams import BiGram, gen_bigrams


def test_gen_bigrams() -> None:
    tokens = [1, 2, 3, 4]
    assert tuple(gen_bigrams(tokens)) == (BiGram(1, 2), BiGram(2, 3), BiGram(3, 4))
