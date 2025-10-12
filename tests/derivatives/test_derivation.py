from regex_utils.derivatives.derivation_rules import derive
from regex_utils.derivatives.components import (Alternation, CharacterSet,
                                            Concatenation, Empty, Epsilon,
                                            KleeneStar, RegexExpr)


def test_derive_concatenation_simple():
    """Test derivation of simple concatenation"""
    # Test ab with respect to 'a'
    a = CharacterSet(["a"])
    b = CharacterSet(["b"])
    c = CharacterSet(["c"])
    concat = Concatenation([a, b, c])

    # a⁻¹(abc) = εbc = bc
    result = derive(concat, "a")
    assert isinstance(result, Concatenation)
    assert len(result.expressions) == 2
    assert isinstance(result.expressions[0], CharacterSet)
    assert result.expressions[0].characters == {"b"}
    assert isinstance(result.expressions[1], CharacterSet)
    assert result.expressions[1].characters == {"c"}

    # f⁻¹(abc) = ∅
    result = derive(concat, "f")
    assert isinstance(result, Empty)


def test_derive_concatenation_nullable():
    """Test derivation of concatenation with nullable first part"""
    # Test a*b with respect to 'b'
    a_star = KleeneStar(CharacterSet(["a"]))
    b = CharacterSet(["b"])
    concat = Concatenation([a_star, b])

    # b⁻¹(a*b) = ε (because a* is nullable)
    result = derive(concat, "b")
    assert isinstance(result, Epsilon)

    # a⁻¹(a*b) = a*b (because a* is nullable)
    result = derive(concat, "a")
    result = result
    print(result)
    print(type(result))

    assert isinstance(result, Concatenation)
    assert len(result.expressions) == 2
    assert isinstance(result.expressions[0], KleeneStar)
    assert isinstance(result.expressions[1], CharacterSet)


def test_derive_concatenation_multiple():
    """Test derivation of concatenation with multiple parts"""
    # Test abc
    a = CharacterSet(["a"])
    b = CharacterSet(["b"])
    c = CharacterSet(["c"])
    concat = Concatenation([a, b, c])

    # a⁻¹(abc) = bc
    result = derive(concat, "a")
    assert isinstance(result, Concatenation)
    assert len(result.expressions) == 2
    assert str(result) == "bc"

    # b⁻¹(abc) = ∅
    result = derive(concat, "b")
    assert isinstance(result, Empty)


def test_derive_alternation_simple():
    """Test derivation of simple alternation"""
    # Test a|b
    a = CharacterSet(["a"])
    b = CharacterSet(["b"])
    alt = Alternation([a, b])

    # a⁻¹(a|b) = ε
    result = derive(alt, "a")
    assert isinstance(result, Epsilon)

    # b⁻¹(a|b) = ε
    result = derive(alt, "b")
    assert isinstance(result, Epsilon)

    # c⁻¹(a|b) = ∅
    result = derive(alt, "c")
    assert isinstance(result, Empty)


def test_derive_alternation_multiple():
    """Test derivation of alternation with multiple parts"""
    # Test a|b|c
    a = CharacterSet(["a"])
    b = CharacterSet(["b"])
    c = CharacterSet(["c"])
    alt = Alternation([a, b, c])

    # a⁻¹(a|b|c) = ε
    result = derive(alt, "a")
    assert isinstance(result, Epsilon)

    # d⁻¹(a|b|c) = ∅
    result = derive(alt, "d")
    assert isinstance(result, Empty)


def test_derive_complex():
    """Test derivation of complex expression"""
    # Test (a|b)c
    a = CharacterSet(["a"])
    b = CharacterSet(["b"])
    c = CharacterSet(["c"])
    alt = Alternation([a, b])
    concat = Concatenation([alt, c])

    # a⁻¹((a|b)c) = c
    result = derive(concat, "a")
    assert result == c

    # b⁻¹((a|b)c) = c
    result = derive(concat, "b")
    assert result == c

    # c⁻¹((a|b)c) = ∅
    result = derive(concat, "c")
    assert isinstance(result, Empty)
