from regex_utils.derivatives.parser import (
    RegexExpr,
    CharacterRange,
    CharacterSet,
    Concatenation,
    Alternation,
    KleeneStar,
    RangeType,
)

def test_character_range_single():
    """Test that a CharacterRange with same start and end is printed without range operator"""
    range_a = CharacterRange('a', 'a')
    assert str(range_a) == "a"

    # Test negative range with same start and end
    range_not_a = CharacterRange('a', 'a', RangeType.NEGATIVE)
    assert str(range_not_a) == "[^a]"

def test_concatenation():
    """Test concatenation of regex expressions"""
    # Test with single token
    range_az = CharacterRange('a', 'z')
    regex = Concatenation([range_az])
    assert str(regex) == "[a-z]"

    # Test with multiple tokens
    charset = CharacterSet(['x', 'y', 'z'])
    regex = Concatenation([range_az, charset])
    assert str(regex) == "[a-z][xyz]"

def test_alternation():
    """Test alternation of regex expressions"""
    # Create tokens
    range_az = CharacterRange('a', 'z')
    charset = CharacterSet(['0', '1', '2'])
    range_AZ = CharacterRange('A', 'Z')
    
    # Test alternation of multiple expressions
    regex = Alternation([range_az, charset, range_AZ])
    assert str(regex) == "([a-z]|[012]|[A-Z])"

def test_nested_expressions():
    """Test nested regex expressions"""
    # Create inner concatenation
    inner = Concatenation([
        CharacterRange('0', '9'),
        CharacterSet(['_'])
    ])
    
    # Create outer alternation with the inner concatenation
    outer = Alternation([
        CharacterRange('a', 'z'),
        inner,
        CharacterRange('A', 'Z')
    ])
    
    assert str(outer) == "([a-z]|([0-9][_])|[A-Z])"

def test_kleene_star():
    """Test Kleene star operation"""
    # Simple character range
    range_az = CharacterRange('a', 'z')
    star = KleeneStar(range_az)
    assert str(star) == "[a-z]*"
    
    # More complex expression
    concat = Concatenation([
        CharacterRange('a', 'z'),
        CharacterSet(['0', '1'])
    ])
    star = KleeneStar(concat)
    assert str(star) == "([a-z][01])*"

def test_complex_expression():
    """Test a more complex regex expression"""
    # Build (a-z|[0-9]_)*[A-Z]
    inner_alt = Alternation([
        CharacterRange('a', 'z'),
        Concatenation([
            CharacterRange('0', '9'),
            CharacterSet(['_'])
        ])
    ])
    star_part = KleeneStar(inner_alt)
    upper_range = CharacterRange('A', 'Z')
    
    regex = Concatenation([star_part, upper_range])
    assert str(regex) == "(([a-z]|([0-9][_])))*[A-Z]"  # Extra parentheses needed for operator precedence

def test_special_characters_escape():
    """Test regex with special characters"""
    # Test character range with special characters
    special_range = CharacterRange('[', ']', RangeType.NEGATIVE)
    assert str(special_range) == r"[^\[-\]]"
    
    # Test character set with special characters
    special_set = CharacterSet(['$', '^', '*'])
    regex = Concatenation([special_range, special_set])
    assert str(regex) == r"[^\[-\]][\$\*\^]"
    
    # Test Kleene star with special characters
    star_special = KleeneStar(special_set)
    assert str(star_special) == r"[\$\*\^]*"