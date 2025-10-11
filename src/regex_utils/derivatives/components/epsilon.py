from dataclasses import dataclass
from typing import Optional

from regex_utils.derivatives.components.regex_expr import RegexExpr


@dataclass
class Epsilon(RegexExpr):
    """Represents the empty string ε"""

    def __contains__(self, char: str) -> bool:
        return '' == char

    def derive(self, _: str) -> 'RegexExpr':
        from .empty import Empty
        return Empty()

    def is_nullable(self) -> bool:
        return True

    def generate(self) -> Optional[str]:
        return ''

    def __repr__(self) -> str:
        return "ε"
