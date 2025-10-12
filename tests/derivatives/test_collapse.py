

from regex_utils.derivatives.components.character_set import CharacterSet, RangeType
from regex_utils.derivatives.components.negation import Negation


def test_charset_collapse():
    a = CharacterSet({"a"})

    assert a.collapse() == a

def test_simple_negative_charset_collapse():
    a = CharacterSet({"a"}, range_type=RangeType.POSITIVE)
    collapsed_b = Negation(a).collapse()
    negated_b = CharacterSet({"a"}, range_type=RangeType.NEGATIVE)

    assert collapsed_b == negated_b

    b = CharacterSet({"a", "b"}, range_type=RangeType.POSITIVE)
    collapsed_b = Negation(b).collapse()
    negated_b = CharacterSet({"a", "b"}, range_type=RangeType.NEGATIVE)

    assert collapsed_b == negated_b