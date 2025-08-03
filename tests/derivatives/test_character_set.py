from regex_utils.derivatives.parser import CharacterSet

def test_character_set_init():
    """Test CharacterSet initialization with both list and set"""
    # Test with list - duplicates are preserved as provided by user
    chars_list = CharacterSet(['a', 'b', 'c', 'a'])
    assert chars_list.characters == ['a', 'b', 'c', 'a']
    
    # Test with set - set will naturally deduplicate
    chars_set = CharacterSet({'x', 'y', 'z'})
    assert set(chars_set.characters) == {'x', 'y', 'z'}
    
    # Test with mixed case - order is preserved as provided by user
    chars_mixed = CharacterSet(['Z', 'a', 'A', 'z'])
    assert chars_mixed.characters == ['Z', 'a', 'A', 'z']

def test_character_set_special_chars():
    """Test CharacterSet representation with special regex characters"""
    # Test with special characters - order is preserved but they are escaped in the output
    special_chars = CharacterSet(['[', ']', '^', '-', '\\'])
    assert special_chars.characters == ['[', ']', '^', '-', '\\']
    assert str(special_chars) == r'[\[\]\^\-\\]'
    
    # Test with mix of special and normal characters
    mixed_chars = CharacterSet(['a', '[', 'b', ']', 'c'])
    assert mixed_chars.characters == ['a', '[', 'b', ']', 'c']
    assert str(mixed_chars) == r'[a\[b\]c]'
    
    # Test with other regex special characters
    regex_chars = CharacterSet(['.', '*', '+', '?', '|', '(', ')', '{', '}'])
    assert set(regex_chars.characters) == {'.', '*', '+', '?', '|', '(', ')', '{', '}'}
    assert all(c in str(regex_chars) for c in [r'\(', r'\)', r'\+', r'\.', r'\?', r'\{', r'\|', r'\}', r'\*'])

def test_character_set_with_ranges():
    """Test CharacterSet with embedded CharacterRange objects"""
    from regex_utils.derivatives.parser import CharacterRange
    
    # Test with a single range
    simple_range = CharacterRange('a', 'z')
    charset = CharacterSet([simple_range])
    assert charset.characters == [simple_range]
    assert str(charset) == '[a-z]'
    
    # Test with multiple ranges
    az = CharacterRange('a', 'z')
    AZ = CharacterRange('A', 'Z')
    digits = CharacterRange('0', '9')
    charset = CharacterSet([az, AZ, digits])
    assert charset.characters == [az, AZ, digits]
    assert str(charset) == '[a-zA-Z0-9]'
    
    # Test mix of ranges and single characters
    charset = CharacterSet([az, '_', digits])
    assert charset.characters == [az, '_', digits]
    assert str(charset) == '[a-z_0-9]'
    
    # Test range with special characters
    special_range = CharacterRange('.', '?')
    charset = CharacterSet([special_range])
    assert str(charset) == r'[\.-\?]'
