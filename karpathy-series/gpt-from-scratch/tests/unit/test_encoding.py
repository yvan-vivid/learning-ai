"""Training tests"""
from gpt_from_scratch.encoding import CharacterEncoding


def test_character_encoding():
    """Test encoding"""
    encoding = CharacterEncoding.from_character_set({'b', 'z', 'a'})
    assert encoding.encode("zzabz") == [2, 2, 0, 1, 2]
    assert encoding.decode((2, 2, 0, 1, 2)) == "zzabz"
