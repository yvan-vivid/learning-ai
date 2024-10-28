from karpathy_series.makemore.bigrams import BiGram, TriGram


def test_gen_bigrams() -> None:
    assert tuple(BiGram.generate(".", "word")) == (
        BiGram(".", "w"),
        BiGram("w", "o"),
        BiGram("o", "r"),
        BiGram("r", "d"),
        BiGram("d", "."),
    )


def test_gen_trigrams() -> None:
    assert tuple(TriGram.generate(".", "word")) == (
        TriGram((".", "."), "w"),
        TriGram((".", "w"), "o"),
        TriGram(("w", "o"), "r"),
        TriGram(("o", "r"), "d"),
        TriGram(("r", "d"), "."),
    )
