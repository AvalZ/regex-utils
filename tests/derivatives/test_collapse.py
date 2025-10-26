import pytest

from regex_utils.derivatives.components.alternation import Alternation
from regex_utils.derivatives.components.character_set import CharacterSet, RangeType
from regex_utils.derivatives.components.concatenation import Concatenation
from regex_utils.derivatives.components.intersection import Intersection
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


def test_concatenation_collapse():
    # Collapse is not defined for concatenation, so this should raise NotImplementedError
    a = CharacterSet({"a"})
    b = CharacterSet({"b"})
    concat = Concatenation([a, b])

    with pytest.raises(NotImplementedError):
        concat.collapse()

def test_intersection_collapse():
    a = CharacterSet({"a", "b", "c"})
    b = CharacterSet({"b", "c", "d"})
    intersection = Intersection([a, b])

    collapsed = intersection.collapse()
    expected = CharacterSet({"b", "c"})

    assert collapsed == expected

def test_intersection_with_negative_charset_collapse():
    a = CharacterSet({"a", "b", "c"})
    b = CharacterSet({"b"}, range_type=RangeType.NEGATIVE)
    intersection = Intersection([a, b])

    collapsed = intersection.collapse()
    expected = CharacterSet({"a", "c"})

    assert collapsed == expected

def test_collapse_intersection_with_negative_charset_only():
    a = CharacterSet({"a"}, range_type=RangeType.NEGATIVE)
    b = CharacterSet({"b"}, range_type=RangeType.NEGATIVE)
    intersection = Intersection([a, b])

    collapsed = intersection.collapse()
    expected = CharacterSet({"a", "b"}, range_type=RangeType.NEGATIVE).get_explicit_charset()

    assert collapsed == expected

def test_collapse_alternation():
    a = CharacterSet({"a"})
    b = CharacterSet({"b"})
    alternation = Alternation([a, b])

    collapsed = alternation.collapse()
    expected = CharacterSet({"a", "b"})

    assert collapsed == expected

def test_collapse_alternation_with_negative_charset():

    custom_alphabet = set("abcdefghijklmnopqrstuvwxyz")

    a = CharacterSet({"a"})
    b = CharacterSet({"b"}, range_type=RangeType.NEGATIVE)
    alternation = Alternation([a, b])

    collapsed = alternation.collapse(custom_alphabet=custom_alphabet)
    expected = CharacterSet(custom_alphabet - {"b"})

    assert collapsed == expected