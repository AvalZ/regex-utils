"""Compatibility layer: re-export component classes and common constants.

This module previously contained all component implementations. They have been
moved to ``regex_utils.derivatives.components`` for better modularity. To
maintain backwards compatibility we import and re-export the public names
here so existing imports continue to work.
"""

from regex_utils.derivatives.components.regex_expr import SPECIAL_CHARS, ALPHABET, RangeType, RegexExpr
from regex_utils.derivatives.components import (
    CharacterSet,
    Empty,
    Epsilon,
    Concatenation,
    Alternation,
    Intersection,
    KleeneStar,
)

__all__ = [
    "SPECIAL_CHARS",
    "ALPHABET",
    "RangeType",
    "RegexExpr",
    "CharacterSet",
    "Empty",
    "Epsilon",
    "Concatenation",
    "Alternation",
    "Intersection",
    "KleeneStar",
]
