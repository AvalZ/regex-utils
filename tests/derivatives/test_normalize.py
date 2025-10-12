from regex_utils.derivatives.parser import (Alternation, CharacterSet,
                                            Concatenation, Empty, Epsilon,
                                            KleeneStar, RangeType, RegexExpr)


def test_normalize_empty():
    """Test normalization of Empty"""
    empty = Empty()

    normalized = empty.normalize()
    assert isinstance(normalized, Empty)


def test_normalize_epsilon():
    """Test normalization of Epsilon"""
    epsilon = Epsilon()

    normalized = epsilon.normalize()
    assert isinstance(normalized, Epsilon)


def test_normalize_alternation_with_only_one_expression():
    """Test normalization of alternation with only one expression"""
    a = CharacterSet(["a"])
    alt = Alternation([a])

    normalized = alt.normalize()
    assert normalized == a


def test_normalize_alternation_with_empty():
    """Test normalization of alternation with Empty"""
    a = CharacterSet(["a"])
    empty = Empty()
    alt1 = Alternation([empty, a])

    normalized1 = alt1.normalize()
    assert normalized1 == a

    alt2 = Alternation([a, empty])

    normalized2 = alt2.normalize()
    assert normalized2 == a


def test_normalize_alernation_with_multiple_empties():
    """Test normalization of alternation with multiple Empties"""
    a = CharacterSet(["a"])
    b = CharacterSet(["b"])
    empty = Empty()
    alt = Alternation([empty, a, empty, b, empty])

    normalized = alt.normalize()
    assert isinstance(normalized, Alternation)
    assert len(normalized.expressions) == 2
    assert normalized.expressions[0] == a
    assert normalized.expressions[1] == b


def test_normalize_alternation_with_epsilon():
    """Test normalization of alternation with Epsilon"""
    a = CharacterSet(["a"])
    epsilon = Epsilon()
    alt1 = Alternation([epsilon, a])

    normalized1 = alt1.normalize()
    assert normalized1 == a

    alt2 = Alternation([a, epsilon])

    normalized2 = alt2.normalize()
    assert normalized2 == a

    b = CharacterSet(["b"])
    alt3 = Alternation([a, epsilon, b])
    normalized3 = alt3.normalize()
    assert isinstance(normalized3, Alternation)
    assert len(normalized3.expressions) == 2
    assert normalized3.expressions[0] == a
    assert normalized3.expressions[1] == b


def test_normalize_alternation_with_all_epsilon():
    """Test normalization of alternation with all Epsilon"""
    epsilon1 = Epsilon()
    epsilon2 = Epsilon()
    alt = Alternation([epsilon1, epsilon2])

    normalized = alt.normalize()
    assert isinstance(normalized, Epsilon)


def test_normalize_nested_alternations():
    """Test normalization of nested alternations"""
    a = CharacterSet(["a"])
    b = CharacterSet(["b"])
    c = CharacterSet(["c"])
    inner_alt = Alternation([b, c])
    outer_alt = Alternation([a, inner_alt])
    normalized = outer_alt.normalize()
    assert isinstance(normalized, Alternation)
    assert len(normalized.expressions) == 3
    assert normalized.expressions[0] == a
    assert normalized.expressions[1] == b
    assert normalized.expressions[2] == c


def test_normalize_multiply_nested_alternations():
    """Test normalization of multiply nested alternations"""
    a = CharacterSet(["a"])
    b = CharacterSet(["b"])
    c = CharacterSet(["c"])
    d = CharacterSet(["d"])
    inner_alt1 = Alternation([c, d])
    inner_alt2 = Alternation([b, inner_alt1])
    outer_alt = Alternation([a, inner_alt2])
    normalized = outer_alt.normalize()
    print(normalized)

    assert isinstance(normalized, Alternation)
    assert len(normalized.expressions) == 4
    assert normalized.expressions[0] == a
    assert normalized.expressions[1] == b
    assert normalized.expressions[2] == c
    assert normalized.expressions[3] == d


def test_normalize_concatenation_with_only_one_expression():
    """Test normalization of concatenation with only one expression"""
    a = CharacterSet(["a"])
    concat = Concatenation([a])

    normalized = concat.normalize()
    assert normalized == a


def test_normalize_concatenation_with_epsilon():
    """Test normalization of concatenation with Epsilon"""
    a = CharacterSet(["a"])
    epsilon = Epsilon()
    concat1 = Concatenation([epsilon, a])

    normalized1 = concat1.normalize()

    assert normalized1 == a


def test_normalize_concatenation_with_empty():
    """Test normalization of concatenation with Empty"""
    a = CharacterSet(["a"])
    empty = Empty()
    concat1 = Concatenation([empty, a])

    normalized1 = concat1.normalize()
    assert isinstance(normalized1, Empty)
