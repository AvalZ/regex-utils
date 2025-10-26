import random
from dataclasses import dataclass
from typing import List, Optional

from regex_utils.derivatives.components.character_set import CharacterSet

from .regex_expr import RegexExpr
from .concatenation import Concatenation
from .epsilon import Epsilon
from .empty import Empty


@dataclass
class Alternation(RegexExpr):
    """Represents a union (|) of regex expressions"""

    expressions: List[RegexExpr]

    def __contains__(self, char: str) -> bool:
        return any(char in expr for expr in self.expressions)

    def peek(self) -> "RegexExpr":
        if not self.expressions:
            return Epsilon()

        peeked_expr = Alternation([])

        for expr in self.expressions:
            peeked_expr.expressions.append(expr.peek())

        return peeked_expr.normalize()

    def derive(self, char: str) -> "RegexExpr":
        from .empty import Empty

        derived_expressions = [expr.derive(char) for expr in self.expressions]
        non_empty_expressions = [
            d for d in derived_expressions if not isinstance(d, Empty)
        ]
        if not non_empty_expressions:
            return Empty()
        elif len(non_empty_expressions) == 1:
            return non_empty_expressions[0]
        return Alternation(non_empty_expressions)

    def is_nullable(self) -> bool:
        return any(expr.is_nullable() for expr in self.expressions)

    def normalize(self) -> "RegexExpr":
        if not self.expressions:
            return Epsilon()

        if len(self.expressions) == 1:
            return self.expressions[0].normalize()

        # Normalize each expression and flatten nested Alternations
        normalized_expressions: list[RegexExpr] = []
        for expr in self.expressions:
            e = expr.normalize()
            if isinstance(e, Alternation):
                normalized_expressions.extend(e.expressions)
            else:
                normalized_expressions.append(e)

        # Remove Empty expressions
        normalized_expressions = [
            expr for expr in normalized_expressions if not isinstance(expr, Empty)
        ]

        if not normalized_expressions:
            return Epsilon()

        # If every remaining expression is Epsilon, return Epsilon
        if all(isinstance(expr, Epsilon) for expr in normalized_expressions):
            return Epsilon()

        # If there is at least one non-Epsilon expression, drop any Epsilon entries
        non_eps = [
            expr for expr in normalized_expressions if not isinstance(expr, Epsilon)
        ]
        if non_eps:
            normalized_expressions = non_eps

        if len(normalized_expressions) == 1:
            return normalized_expressions[0]

        return Alternation(normalized_expressions)

    def __repr__(self) -> str:
        parts = []
        for expr in self.expressions:
            if isinstance(expr, Concatenation) and len(expr.expressions) > 1:
                parts.append(f"({expr})")
            else:
                parts.append(str(expr))
        return f"({('|'.join(parts))})"

    def collapse(self, custom_alphabet: set[str] | None = None) -> CharacterSet:
        if not self.expressions:
            return CharacterSet(set())

        # Ensure any negative character sets are expanded to explicit positive
        # character lists before taking the union (consistent with Intersection)
        characters = [
            expr.collapse(custom_alphabet=custom_alphabet).get_explicit_charset().characters for expr in self.expressions
        ]

        union_character_sets = set.union(*characters) if characters else set()

        return CharacterSet(union_character_sets)

