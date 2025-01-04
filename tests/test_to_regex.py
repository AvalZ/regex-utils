from regex_utils import regex


def test_to_regex():
    assert "hello" == regex.from_string("hello").to_string()
    assert "a*bcd" == regex.from_string("a*bcd").to_string()


def test_intersection():
    assert "abc" == regex.intersect("ab[a-z]", "abc*").to_string()


def test_to_dot():
    # regex.from_string("ab[a-z]d*").to_dot()
    pass
