from dataclasses import dataclass
from typing import List, Optional

from regex_utils.derivatives.components.regex_expr import RegexExpr


@dataclass
class Concatenation(RegexExpr):
    """Represents a concatenation of regex expressions"""

    expressions: List[RegexExpr]

    def __contains__(self, char: str) -> bool:
        return char in self.expressions[0]

    def peek(self) -> RegexExpr:
        from .alternation import Alternation

        if not self.expressions:
            from .epsilon import Epsilon

            return Epsilon()

        peeked_expr = Alternation([])

        first_expr, *remaining_expr = self.expressions
        peeked_expr.expressions.append(first_expr.peek())

        while first_expr.is_nullable() and remaining_expr:
            peeked_expr.expressions.append(remaining_expr[0].peek())
            first_expr, *remaining_expr = remaining_expr

        return peeked_expr.normalize()

    def derive(self, char: str) -> "RegexExpr":
        from .alternation import Alternation
        from .concatenation import Concatenation as ConcatCls

        if not self.expressions:
            from .empty import Empty

            return Empty()

        derived_expr = Alternation([])
        first_expr, *remaining_expr = self.expressions

        while first_expr.is_nullable() and remaining_expr:

            derived_expr.expressions.append(
                ConcatCls([first_expr.derive(char), *remaining_expr]),
            )

            first_expr, *remaining_expr = remaining_expr

        derived_expr.expressions.append(
            ConcatCls([first_expr.derive(char), *remaining_expr]),
        )

        return derived_expr

    def is_nullable(self) -> bool:
        return all(expr.is_nullable() for expr in self.expressions)

    def generate(self, custom_alphabet: Optional[set[str]] = None) -> Optional[str]:
        if not self.expressions:
            return ""

        return self.expressions[0].generate(custom_alphabet=custom_alphabet)

    def normalize(self) -> "RegexExpr":
        from .empty import Empty
        from .epsilon import Epsilon

        if len(self.expressions) == 1:
            return self.expressions[0].normalize()

        if not self.expressions:
            return Epsilon()

        first_expression = self.expressions[0].normalize()

        if first_expression and isinstance(first_expression, Empty):
            return Empty()

        if first_expression and isinstance(first_expression, Epsilon):
            from .concatenation import Concatenation as ConcatCls

            return ConcatCls(self.expressions[1:]).normalize()

        return Concatenation([first_expression, *self.expressions[1:]])

    def __post_init__(self):
        if len(self.expressions) < 1:
            raise ValueError("Concatenation must have at least one expression")

    def __repr__(self) -> str:
        parts = []
        for expr in self.expressions:
            if isinstance(
                expr,
                (
                    Alternation := __import__(
                        "regex_utils.derivatives.components.alternation",
                        fromlist=["Alternation"],
                    ).Alternation
                ),
            ):
                parts.append(str(expr))
            else:
                parts.append(str(expr))
        return "".join(parts)
