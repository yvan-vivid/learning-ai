from karpathy_series.makemore.encoding.word_encoding import DelimitedWordEncoder

TEST_TOKENS = frozenset(("a", "b", "c"))
TEST_ENCODING = DelimitedWordEncoder.from_charset(TEST_TOKENS)
TEST_WORD_ENCODING = TEST_ENCODING.word_encoder
TEST_CHARACTER_ENCODING = TEST_WORD_ENCODING.letter_encoder

EN_CASES = (("a", 0), ("b", 1), ("c", 2))


def test_encode() -> None:
    for t, e in EN_CASES:
        assert TEST_CHARACTER_ENCODING.encode(t) == e


def test_encode_none() -> None:
    assert TEST_CHARACTER_ENCODING.encode("d") is None


def test_encode_word() -> None:
    assert TEST_WORD_ENCODING.encode("bcca") == [1, 2, 2, 0]


def test_encode_delimited_word() -> None:
    assert TEST_ENCODING.encode("bcca") == [TEST_ENCODING.boundary, 1, 2, 2, 0, TEST_ENCODING.boundary]


def test_decode() -> None:
    for t, e in EN_CASES:
        assert TEST_CHARACTER_ENCODING.decode(e) == t


def test_decode_none() -> None:
    assert TEST_CHARACTER_ENCODING.decode(5) is None


def test_decode_word() -> None:
    assert TEST_WORD_ENCODING.decode([1, 2, 2, 0]) == "bcca"


def test_decode_delimited_word() -> None:
    assert TEST_ENCODING.decode([TEST_ENCODING.boundary, 1, 2, 2, 0, TEST_ENCODING.boundary]) == "bcca"
