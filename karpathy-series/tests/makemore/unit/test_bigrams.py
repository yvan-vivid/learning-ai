from karpathy_series.makemore.bigrams import BiGram, NGram, TriGram


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


def test_gen_ngrams() -> None:
    assert tuple(NGram.generate(3, ".", "word")) == (
        NGram((".", ".", "."), "w"),
        NGram((".", ".", "w"), "o"),
        NGram((".", "w", "o"), "r"),
        NGram(("w", "o", "r"), "d"),
        NGram(("o", "r", "d"), "."),
    )

    assert tuple(NGram.generate(4, ".", "word")) == (
        NGram((".", ".", ".", "."), "w"),
        NGram((".", ".", ".", "w"), "o"),
        NGram((".", ".", "w", "o"), "r"),
        NGram((".", "w", "o", "r"), "d"),
        NGram(("w", "o", "r", "d"), "."),
    )
