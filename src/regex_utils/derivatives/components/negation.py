from dataclasses import dataclass
from typing import Optional

from regex_utils.derivatives.components import (Alternation, Concatenation,
                                                Empty)
from regex_utils.derivatives.components.character_set import CharacterSet
from regex_utils.derivatives.components.regex_expr import RegexExpr


@dataclass
class Negation(RegexExpr):
    """Represents a negation operation"""

    expression: RegexExpr

    def __contains__(self, char: str) -> bool:
        return char not in self.expression

    def peek(self) -> RegexExpr:
        return Negation(self.expression.peek())
    
    def collapse(self) -> CharacterSet:
        return self.expression.collapse().negate()

    def derive(self, char: str) -> "RegexExpr":
        d_expr = self.expression.derive(char)
        if isinstance(d_expr, Empty):
            return Empty()
        return Concatenation([d_expr, self])

    def is_nullable(self) -> bool:
        return True

    def __repr__(self) -> str:
        # Negation has higher precedence than alternation but lower than grouping
        # We need parentheses for alternation or concatenation with multiple elements
        needs_parens = isinstance(self.expression, Alternation) or (
            isinstance(self.expression, Concatenation)
            and len(self.expression.expressions) > 1
        )
        base = str(self.expression)
        return f"¬({base})" if needs_parens else f"¬{base}"
