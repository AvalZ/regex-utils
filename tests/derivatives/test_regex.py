from regex_utils.derivatives.parser import (Alternation, CharacterSet,
                                            Concatenation, KleeneStar,
                                            RangeType, RegexExpr)


def test_character_set_single():
    """Test CharacterSet with single characters"""
    char_a = CharacterSet(["a"])
    assert str(char_a) == "[a]"

    # Test negative set with single character
    not_a = CharacterSet(["a"], range_type=RangeType.NEGATIVE)
    assert str(not_a) == "[^a]"


def test_concatenation():
    """Test concatenation of regex expressions"""
    # Test with single token
    letters = CharacterSet(["a", "b", "c", "d", "e", "f"])
    regex = Concatenation([letters])
    assert str(regex) == "[abcdef]"

    # Test with multiple tokens
    charset = CharacterSet(["x", "y", "z"])
    regex = Concatenation([letters, charset])
    assert str(regex) == "[abcdef][xyz]"


def test_alternation():
    """Test alternation of regex expressions"""
    # Create tokens
    lower = CharacterSet(["a", "b", "c"])
    digits = CharacterSet(["0", "1", "2"])
    upper = CharacterSet(["X", "Y", "Z"])

    # Test alternation of multiple expressions
    regex = Alternation([lower, digits, upper])
    assert str(regex) == "([abc]|[012]|[XYZ])"


def test_nested_expressions():
    """Test nested regex expressions"""
    # Create inner concatenation
    inner = Concatenation([CharacterSet(["0", "1", "2"]), CharacterSet(["_"])])

    # Create outer alternation with the inner concatenation
    outer = Alternation(
        [CharacterSet(["a", "b", "c"]), inner, CharacterSet(["X", "Y", "Z"])]
    )

    assert str(outer) == "([abc]|([012][_])|[XYZ])"


def test_kleene_star():
    """Test Kleene star operation"""
    # Simple character set
    letters = CharacterSet(["a", "b", "c"])
    star = KleeneStar(letters)
    assert str(star) == "[abc]*"

    # More complex expression
    concat = Concatenation([CharacterSet(["a", "b", "c"]), CharacterSet(["0", "1"])])
    star = KleeneStar(concat)
    assert str(star) == "([abc][01])*"


def test_complex_expression():
    """Test a more complex regex expression"""
    # Build ([abc]|[012]_)*[XYZ]
    inner_alt = Alternation(
        [
            CharacterSet(["a", "b", "c"]),
            Concatenation([CharacterSet(["0", "1", "2"]), CharacterSet(["_"])]),
        ]
    )
    star_part = KleeneStar(inner_alt)
    upper = CharacterSet(["X", "Y", "Z"])

    regex = Concatenation([star_part, upper])
    assert str(regex) == "(([abc]|([012][_])))*[XYZ]"


def test_special_characters_escape():
    """Test regex with special characters"""
    # Test negative character set with special characters
    special_set = CharacterSet(["[", "]"], range_type=RangeType.NEGATIVE)
    assert str(special_set) == r"[^\[\]]"

    # Test character set with special characters
    special_set2 = CharacterSet(["$", "^", "*"])
    regex = Concatenation([special_set, special_set2])
    assert str(regex) == r"[^\[\]][\$\^\*]"

    # Test Kleene star with special characters
    star_special = KleeneStar(special_set)
    assert str(star_special) == r"[\$\*\^]*"
