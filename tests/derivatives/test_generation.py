import random

from regex_utils.derivatives.derivation_rules import derive
from regex_utils.derivatives.parser import (CharacterSet, Concatenation,
                                            Epsilon, KleeneStar, RangeType)


def test_generation_from_charset():
    """Test generation of regex from CharacterSet"""
    random.seed(0)  # For reproducibility

    charset = CharacterSet(["a", "b", "c"])

    assert charset.generate() in ["a", "b", "c"]

    assert charset.generate() == "b"


def test_generation_from_negative_charset():
    """Test generation of regex from negative CharacterSet"""

    negative_charset = CharacterSet(["a", "b", "c"], range_type=RangeType.NEGATIVE)
    custom_alphabet = set("abcdef")

    assert negative_charset.generate(custom_alphabet=custom_alphabet) in ["d", "e", "f"]


def test_generation_from_empty_charset():
    """Test generation from an empty CharacterSet"""
    random.seed(0)

    empty_charset = CharacterSet([])

    assert empty_charset.generate() is None


def test_generation_from_negative_empty_charset():
    """Test generation from a negative CharacterSet"""
    random.seed(0)  # For reproducibility
    negative_empty_charset = CharacterSet([], range_type=RangeType.NEGATIVE)
    custom_alphabet = set("abcdef")

    assert negative_empty_charset.generate(custom_alphabet=custom_alphabet) in [
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
    ]


def test_generation_from_epsilon():
    """Test generation from Epsilon"""
    random.seed(0)

    epsilon = Epsilon()

    assert epsilon.generate() == ""


def test_generation_from_concatenation():
    """Test generation from Concatenation"""

    a = CharacterSet(["a"])
    b = CharacterSet(["b"])
    c = CharacterSet(["c"])
    concat = Concatenation([a, b, c])

    assert concat.generate() == "a"


def test_generation_from_alternation(): ...


def test_generation_from_intersection(): ...


def test_generation_from_kleene_star():
    a = CharacterSet(["a"])
    a_star = KleeneStar(a)

    assert a_star.generate() == "a"
