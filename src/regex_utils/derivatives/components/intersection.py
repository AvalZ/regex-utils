from dataclasses import dataclass
from typing import List

from regex_utils.derivatives.components.character_set import CharacterSet
from regex_utils.derivatives.components.epsilon import Epsilon
from regex_utils.derivatives.components.regex_expr import RegexExpr


@dataclass
class Intersection(RegexExpr):
    """Represents an intersection (&) of regex expressions"""

    expressions: List[RegexExpr]

    def __contains__(self, char: str) -> bool:
        return all(char in expr for expr in self.expressions)

    def is_nullable(self) -> bool:
        return all(expr.is_nullable() for expr in self.expressions)

    def peek(self) -> "RegexExpr":
        if not self.expressions:
            return Epsilon()

        return Intersection([expr.peek() for expr in self.expressions]).normalize()

    def __repr__(self) -> str:
        return " ∩ ".join(repr(expr) for expr in self.expressions)

    def derive(self, char: str) -> RegexExpr:
        raise NotImplementedError

    def generate(self, custom_alphabet: set[str] | None = None) -> str | None:
        raise NotImplementedError

    def collapse(self) -> CharacterSet:
        if not self.expressions:
            return CharacterSet(set())

        # FIXME get_explicit_charset is never used with a custom_alphabet here
        characters = [expr.collapse().get_explicit_charset().characters for expr in self.expressions]
        intersected_chars = set.intersection(*characters)

        return CharacterSet(intersected_chars)