from regex_utils.derivatives.components import CharacterSet, RangeType, Concatenation
from regex_utils.derivatives.parser.regex_parser import RegexParser

def test_parse_literal():

    parser = RegexParser("a")
    regex_expr = parser.parse()
    assert regex_expr == CharacterSet(["a"])

def test_parse_negated_literal():
    
    parser = RegexParser("[^a]")
    regex_expr = parser.parse()
    assert regex_expr == CharacterSet(["a"], range_type=RangeType.NEGATIVE)

def test_parse_range():
    
    parser = RegexParser("[abc]")
    regex_expr = parser.parse()
    assert regex_expr == CharacterSet(["a", "b", "c"])

def test_parse_negated_range():
    
    parser = RegexParser("[^abc]")
    regex_expr = parser.parse()
    assert regex_expr == CharacterSet(["a", "b", "c"], range_type=RangeType.NEGATIVE)

def test_parse_concatenation():
    
    parser = RegexParser("fbc")
    regex_expr = parser.parse()
    expected = Concatenation([CharacterSet(["f"]), CharacterSet(["b"]), CharacterSet(["c"])])
    assert regex_expr == expected


def test_complex_expression():
    
    parser = RegexParser("a[^bcd]c")
    regex_expr = parser.parse()
    expected = Concatenation([
        CharacterSet(["a"]),
        CharacterSet(["b", "c", "d"], range_type=RangeType.NEGATIVE),
        CharacterSet(["c"])
    ])
    assert regex_expr == expected