from regex_utils.derivatives.parser import CharacterSet, RangeType


def test_character_set_init():
    """Test CharacterSet initialization with both list and set"""
    # Test with list - duplicates are preserved as provided by user
    chars_list = CharacterSet(["a", "b", "c", "a"])
    assert chars_list.characters == ["a", "b", "c", "a"]

    # Test with set - set will naturally deduplicate
    chars_set = CharacterSet({"x", "y", "z"})
    assert set(chars_set.characters) == {"x", "y", "z"}

    # Test with mixed case - order is preserved as provided by user
    chars_mixed = CharacterSet(["Z", "a", "A", "z"])
    assert chars_mixed.characters == ["Z", "a", "A", "z"]


def test_character_set_special_chars():
    """Test CharacterSet representation with special regex characters"""
    # Test with special characters - order is preserved but they are escaped in the output
    special_chars = CharacterSet(["[", "]", "^", "-", "\\"])
    assert special_chars.characters == ["[", "]", "^", "-", "\\"]
    assert str(special_chars) == r"[\[\]\^\-\\]"

    # Test with mix of special and normal characters
    mixed_chars = CharacterSet(["a", "[", "b", "]", "c"])
    assert mixed_chars.characters == ["a", "[", "b", "]", "c"]
    assert str(mixed_chars) == r"[a\[b\]c]"

    # Test with other regex special characters
    regex_chars = CharacterSet([".", "*", "+", "?", "|", "(", ")", "{", "}"])
    assert set(regex_chars.characters) == {".", "*", "+", "?", "|", "(", ")", "{", "}"}
    assert all(
        c in str(regex_chars)
        for c in [r"\(", r"\)", r"\+", r"\.", r"\?", r"\{", r"\|", r"\}", r"\*"]
    )


def test_character_set_with_multichars():
    """Test CharacterSet with characters that span a range"""
    # Test lowercase letters
    charset = CharacterSet(["a", "b", "c", "d", "e"])
    assert charset.characters == ["a", "b", "c", "d", "e"]
    assert str(charset) == "[abcde]"

    # Test mixed case and digits
    charset = CharacterSet(["a", "b", "c", "A", "B", "C", "0", "1", "2"])
    assert charset.characters == ["a", "b", "c", "A", "B", "C", "0", "1", "2"]
    assert str(charset) == "[abcABC012]"

    # Test with underscore and digits
    charset = CharacterSet(["a", "b", "c", "_", "0", "1", "2"])
    assert charset.characters == ["a", "b", "c", "_", "0", "1", "2"]
    assert str(charset) == "[abc_012]"

    # Test with special characters
    charset = CharacterSet([".", "/", "?"])
    # FIXME: escaping ? is not really needed in [], might be overkill in this case
    assert str(charset) == r"[\.\/\?]"


def test_contains_for_charset():

    charset = CharacterSet(["a", "b", "c"])
    assert "a" in charset
    assert "d" not in charset


def test_contains_for_negative_charset():

    negative_charset = CharacterSet(["a", "b", "c"], range_type=RangeType.NEGATIVE)
    assert "a" not in negative_charset
    assert "d" in negative_charset
