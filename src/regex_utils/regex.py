from nfa import NFA


class Regex:
    def __init__(self, regex_string=None, nfa=None):
        self.regex_string = regex_string
        self.nfa = nfa

        if regex_string and not nfa:
            self.nfa = NFA.from_regex(regex_string).simplify()
        elif nfa and not regex_string:
            self.regex_string = nfa.to_regex()

    def __str__(self):
        return self.to_string()

    def to_string(self):
        if self.regex_string is None:
            self.regex_string = self.nfa.to_regex()

        return self.regex_string

    def to_dot(self, view=False):
        return self.nfa.to_dot(view=view)

    def generate_sample(self):
        """
        Generate a string that matches the regex

        Example:

            from Regex import regex
            r = Regex("[a-z]{5}")
            print(r.generate_sample())
            # Output: "xfkdy"

            samples = [r.generate_sample() for _ in range(4)]
            # samples: ['yvegt', 'bqxne', 'wvlmi', 'osieh']

        :return:
        """
        sample = self.nfa.walk()

        return sample


def from_nfa(nfa):
    """
    Create a regex from an NFA
    Equivalent to `Regex(nfa=nfa)`

    :param nfa: an NFA object
    :return: a Regex object
    """
    return Regex(nfa=nfa)


def from_string(regex_string):
    """
    Create a regex from a string
    Equivalent to `Regex(regex_string=regex_string)`

    Example:

        import regex
        regex.from_string("[a-z]{5}").generate_sample()
        # Output: "xfkdy"

    :param regex_string: a regex string
    :return: a Regex object
    """
    return Regex(regex_string=regex_string)


def negate(r):
    """
    Negate a regex

    :param r: a regex string or a Regex object
    :return: a Regex object that matches everything except the input regex
    """

    if type(r) is str:
        r = Regex(regex_string=r)

    negated_nfa = NFA.negate(r.nfa)
    negated_regex = Regex(nfa=negated_nfa)

    return negated_regex


def intersect(*regexes):
    """
    Intersect multiple regexes

    :param regexes: a list of regex strings or Regex objects
    :return: resulting regex
    """
    regexes = [Regex(r) if type(r) == str else r for r in regexes]
    nfas = [r.nfa for r in regexes]

    intersection = nfas[0].simplify()
    for nfa in nfas[1:]:
        intersection = NFA.intersect(intersection, nfa).simplify()

    return Regex(nfa=intersection)


if __name__ == "__main__":
    r = intersect(
        "abc*d",
        "\w{5}",
        "[a-f]{5}"
    )

    r = negate(r)
    print(r.to_string())
    print(r.to_dot(view=True))

    print(r.generate_sample())
    print(r.generate_samples(5))
