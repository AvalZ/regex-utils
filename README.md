# Regex Utils
A set of utils to work with Regular Expressions in Python

## Main Features
- Generate strings that match a given regex
- Intersect two regular expressions
- Negate a regular expression (experimental)
- Convert a NFA back into a regex

Use Python native `sre_parse` to parse a regex, then convert it into a [Nondeterministic Finite Automaton](https://en.wikipedia.org/wiki/Nondeterministic_finite_automaton)

Once you have this NFA, you can use all the features in this package. The `regex` package offers an abstraction for NFA-based operations.

## String Generation

Generate strings that match the given regex, by performing a random walk over the generated NFA.

## Intersection

Intersect two regular expressions.

This is the equivalent of having to match both regex, one after the other. The advantage is that you can now use the `to_string` function to generate the resulting regular expression.

This is also useful if you want to “compile” lookarounds into the regex itself.

Finally, you can use intersection to generate strings of arbitrary length.
For example, if you had the `ab+c*` regex, and you wanted a minimum length of 3 characters and a maximum length of 10 character for your string, you could intersect the regex like this:

```
regex.intersect("ab+c*", ".{3,10}")
```
You could then generate strings using the resulting query.

Please notice that Intersections could result in empty queries. For example, if you tried to intersect two regex with nothing in common, you would receive an empty regex as a result.

```
regex.intersect("[a-z]", "[^a-z]")
```

## Negation (experimental)

Complement the original NFA by converting accepting states to non-accepting states, and add all missing transitions (this is used to generate random strings). The logic is similar to a Negative Lookahead in PCRE.

WARNING: this feature is currently experimental, and it contains some known bugs for specific scenarios.

Please open a issue if you find any specific bugs related to this feature 🙏

## To String

Once you have the resulting NFA, you can get the original regex back in plain text, so that you can use it in other tools.
