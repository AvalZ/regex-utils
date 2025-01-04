from regex_utils import regex


def test_word_boundaries_partial_nfa():
    r = regex.from_string("cb*f?a*\\bbc")
    nfa = r.regex_nfa

    nfa._get_partial_nfa_for_word_boundary(nfa.boundaries[0].from_state).to_dot(
        view=True
    )
