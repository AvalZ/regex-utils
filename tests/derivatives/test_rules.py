from regex_utils.derivatives.parser import CharacterRange, RangeType

def test_character_range_repr_positive():
    """Test string representation of positive character ranges"""
    # Test lowercase range
    range_az = CharacterRange('a', 'z')
    assert str(range_az) == "[a-z]"
    
    # Test uppercase range
    range_AZ = CharacterRange('A', 'Z')
    assert str(range_AZ) == "[A-Z]"
    
    # Test numeric range
    range_09 = CharacterRange('0', '9')
    assert str(range_09) == "[0-9]"
    
    # Test custom range
    range_fk = CharacterRange('f', 'k')
    assert str(range_fk) == "[f-k]"
    
    # Test range with special characters
    range_special = CharacterRange('[', ']')
    assert str(range_special) == r"[\[-\]]"
    
    # Test range with backslash
    range_escape = CharacterRange('\\', '^')
    assert str(range_escape) == r"[\\-\^]"

def test_character_range_repr_negative():
    """Test string representation of negative character ranges"""
    # Test lowercase range
    range_not_az = CharacterRange('a', 'z', RangeType.NEGATIVE)
    assert str(range_not_az) == "[^a-z]"
    
    # Test uppercase range
    range_not_AZ = CharacterRange('A', 'Z', RangeType.NEGATIVE)
    assert str(range_not_AZ) == "[^A-Z]"
    
    # Test numeric range
    range_not_09 = CharacterRange('0', '9', RangeType.NEGATIVE)
    assert str(range_not_09) == "[^0-9]"
    
    # Test custom range
    range_not_fk = CharacterRange('f', 'k', RangeType.NEGATIVE)
    assert str(range_not_fk) == "[^f-k]"
    
    # Test negative range with special characters
    range_not_special = CharacterRange('[', ']', RangeType.NEGATIVE)
    assert str(range_not_special) == r"[^\[-\]]"
    
    # Test negative range with backslash
    range_not_escape = CharacterRange('\\', '^', RangeType.NEGATIVE)
    assert str(range_not_escape) == r"[^\\-\^]"
