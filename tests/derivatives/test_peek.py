from regex_utils.derivatives.components import (Alternation, CharacterSet,
                                            Concatenation, Empty, Epsilon,
                                            KleeneStar, Intersection)


def test_simple_peek():
    assert Epsilon().peek() == Epsilon()
    assert Empty().peek() == Empty()
    assert CharacterSet({"a", "b", "c"}).peek() == CharacterSet({"a", "b", "c"})


def test_peek_concatenation_without_nullable_elements():
    a = CharacterSet({"a"})
    b = CharacterSet({"b"})
    c = CharacterSet({"c"})
    concat = Concatenation([a, b, c])
    assert concat.peek() == a


def test_peek_concatenation_with_nullable_elements():
    a = CharacterSet({"a"})
    b = CharacterSet({"b"})
    c = CharacterSet({"c"})
    concat = Concatenation([KleeneStar(a), b, c])
    peeked_alternatives = concat.peek()
    assert isinstance(peeked_alternatives, Alternation)
    assert len(peeked_alternatives.expressions) == 2
    assert peeked_alternatives.expressions[0] == a
    assert peeked_alternatives.expressions[1] == b

    concat = Concatenation([KleeneStar(a), KleeneStar(b), c])
    peeked_alternatives = concat.peek()
    assert isinstance(peeked_alternatives, Alternation)
    assert len(peeked_alternatives.expressions) == 3
    assert peeked_alternatives.expressions[0] == a
    assert peeked_alternatives.expressions[1] == b
    assert peeked_alternatives.expressions[2] == c

    concat = Concatenation([KleeneStar(a), KleeneStar(b), KleeneStar(c)])
    peeked_alternatives = concat.peek()
    assert isinstance(peeked_alternatives, Alternation)
    assert len(peeked_alternatives.expressions) == 3
    assert peeked_alternatives.expressions[0] == a
    assert peeked_alternatives.expressions[1] == b
    assert peeked_alternatives.expressions[2] == c


def test_peek_alternation():
    a = CharacterSet({"a"})
    b = CharacterSet({"b"})
    c = CharacterSet({"c"})
    alt = Alternation([a, b, c])
    peeked = alt.peek()
    assert isinstance(peeked, Alternation)
    assert len(peeked.expressions) == 3
    assert a in peeked.expressions
    assert b in peeked.expressions
    assert c in peeked.expressions


def test_peek_alternation_with_concatenations():
    a = CharacterSet({"a"})
    b = CharacterSet({"b"})
    c = CharacterSet({"c"})
    d = CharacterSet({"d"})
    concat1 = Concatenation([a, b])
    concat2 = Concatenation([c, d])
    alt = Alternation([concat1, concat2])
    peeked = alt.peek()
    assert isinstance(peeked, Alternation)
    assert len(peeked.expressions) == 2
    assert peeked.expressions[0] == a
    assert peeked.expressions[1] == c

def test_peek_intersection():
    a = CharacterSet({"a"})
    bx = Concatenation([CharacterSet({"b"}), CharacterSet({"x"})])
    c = CharacterSet({"c"})
    d = CharacterSet({"d"})
    intersection = Intersection([a, bx, c, d])
    peeked = intersection.peek()
    assert peeked == Intersection([a, CharacterSet({"b"}), c, d])


def test_peek_complex_expression():
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
    peeked = regex.peek()
    assert isinstance(peeked, Alternation)
    assert len(peeked.expressions) == 3
    assert CharacterSet({"a", "b", "c"}) in peeked.expressions
    assert CharacterSet({"0", "1", "2"}) in peeked.expressions
    assert CharacterSet({"X", "Y", "Z"}) in peeked.expressions


def test_another_complex_expression():
    # Build (a*[xyz]*)efg|[abc]more
    a = CharacterSet({"a"})
    abc = CharacterSet({"a", "b", "c"})
    xyz = CharacterSet({"x", "y", "z"})
    efg = Concatenation([CharacterSet({"e"}), CharacterSet({"f"}), CharacterSet({"g"})])
    more = Concatenation(
        [
            CharacterSet({"m"}),
            CharacterSet({"o"}),
            CharacterSet({"r"}),
            CharacterSet({"e"}),
        ]
    )

    alt = Alternation(
        [
            Concatenation([KleeneStar(a), KleeneStar(xyz), efg]),
            Concatenation([abc, more]),
        ]
    )

    peeked = alt.peek()
    assert isinstance(peeked, Alternation)
    assert len(peeked.expressions) == 4
    assert peeked.expressions[0] == a
    assert peeked.expressions[1] == xyz
    assert peeked.expressions[2] == CharacterSet({"e"})
    assert peeked.expressions[3] == abc
