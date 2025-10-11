from typing import List

from regex_utils.derivatives.components.regex_expr import RegexExpr


class Intersection(RegexExpr):
    """Represents an intersection (&) of regex expressions"""
    expressions: List[RegexExpr]

    def __contains__(self, char: str) -> bool:
        return all(char in expr for expr in self.expressions)

    def is_nullable(self) -> bool:
        return all(expr.is_nullable() for expr in self.expressions)
