from regex_utils import regex


def test_simple_word_boundaries_partial_nfa():
    r = regex.from_string(r".\b.")

    assert r.to_string() == r"(?:(?:\W\w|\w\W))"


def test_consecutive_word_boundaries_partial_nfa():
    r = regex.from_string(r".\b.\b.")

    assert r.to_string() == r"(?:(?:\W\w\W|\w\W\w))"


def test_word_boundary_vs_fixed_char_before():
    r = regex.from_string(r"abc\b.")

    assert r.to_string() == r"abc\W"


def test_word_boundary_vs_fixed_char_after():
    r = regex.from_string(r".\bdef")

    assert r.to_string() == r"\Wdef"


def test_impossible_word_boundary():
    r = regex.from_string(r"a\ba")

    assert r.to_string() == r""


def test_word_boundary_on_repetition():
    r = regex.from_string(r"a\b.*")

    assert r.to_string() == r"a|a\W(?:.)*"


def test_word_boundary_on_alternated_repetitions():
    r = regex.from_string(r"a\b(.*|def)")

    assert r.to_string() == r"a\W.*"
