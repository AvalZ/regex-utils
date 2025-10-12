from regex_utils.derivatives.components import CharacterSet, RangeType


def test_character_set_repr_positive():
    """Test string representation of positive character sets"""
    # Test lowercase set
    set_abc = CharacterSet(["a", "b", "c"])
    assert str(set_abc) == "[abc]"

    # Test uppercase set
    set_ABC = CharacterSet(["A", "B", "C"])
    assert str(set_ABC) == "[ABC]"

    # Test numeric set
    set_012 = CharacterSet(["0", "1", "2"])
    assert str(set_012) == "[012]"

    # Test custom set
    set_fghk = CharacterSet(["f", "g", "h", "k"])
    assert str(set_fghk) == "[fghk]"

    # Test set with special characters
    set_special = CharacterSet(["[", "]"])
    assert str(set_special) == r"[\[\]]"

    # Test set with backslash
    set_escape = CharacterSet(["\\", "^"])
    assert str(set_escape) == r"[\\\^]"


def test_character_set_repr_negative():
    """Test string representation of negative character sets"""
    # Test lowercase set
    set_not_abc = CharacterSet(["a", "b", "c"], RangeType.NEGATIVE)
    assert str(set_not_abc) == "[^abc]"

    # Test uppercase set
    set_not_ABC = CharacterSet(["A", "B", "C"], RangeType.NEGATIVE)
    assert str(set_not_ABC) == "[^ABC]"

    # Test numeric set
    set_not_012 = CharacterSet(["0", "1", "2"], RangeType.NEGATIVE)
    assert str(set_not_012) == "[^012]"

    # Test custom set
    set_not_fghk = CharacterSet(["f", "g", "h", "k"], RangeType.NEGATIVE)
    assert str(set_not_fghk) == "[^fghk]"

    # Test negative set with special characters
    set_not_special = CharacterSet(["[", "]"], RangeType.NEGATIVE)
    assert str(set_not_special) == r"[^\[\]]"

    # Test negative set with backslash
    set_not_escape = CharacterSet(["\\", "^"], RangeType.NEGATIVE)
    assert str(set_not_escape) == r"[^\\\^]"
