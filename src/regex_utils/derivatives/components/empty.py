from dataclasses import dataclass
from typing import Optional

from regex_utils.derivatives.components.regex_expr import RegexExpr


@dataclass
class Empty(RegexExpr):
    """Represents the empty language ∅"""

    def __contains__(self, _: str) -> bool:
        return False

    def derive(self, _: str) -> "RegexExpr":
        return self

    def is_nullable(self) -> bool:
        return False

    def generate(self) -> Optional[str]:
        return None

    def __repr__(self) -> str:
        return "∅"
