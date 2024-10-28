from karpathy_series.makemore.encoding.character import BiCharacterEncoder, CharacterEncoder, CharacterSet

TEST_CHAR_SET = CharacterSet.from_words(("abc",))
TEST_CHAR_ENCODING = CharacterEncoder.from_charset(TEST_CHAR_SET)
TEST_BI_CHAR_ENCODING = BiCharacterEncoder.from_charset(TEST_CHAR_SET)

EN_CASES = (("a", 1), ("b", 2), ("c", 3))
BI_EN_CASES = ((("a", "a"), 5), (("a", "b"), 6), (("a", "c"), 7), (("b", "a"), 9), (("b", "b"), 10))


def test_encode() -> None:
    for t, e in EN_CASES:
        assert TEST_CHAR_ENCODING.encode(t) == e


def test_encode_bi() -> None:
    for t, e in BI_EN_CASES:
        assert TEST_BI_CHAR_ENCODING.encode(t) == e


def test_encode_none() -> None:
    assert TEST_CHAR_ENCODING.encode("d") is None


def test_decode() -> None:
    for t, e in EN_CASES:
        assert TEST_CHAR_ENCODING.decode(e) == t


def test_decode_bi() -> None:
    for t, e in BI_EN_CASES:
        assert TEST_BI_CHAR_ENCODING.decode(e) == t


def test_decode_none() -> None:
    assert TEST_CHAR_ENCODING.decode(5) is None
