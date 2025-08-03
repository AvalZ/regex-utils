

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from typing import List, Union, Optional


# Characters that need to be escaped in regular expressions
SPECIAL_CHARS = {']', '[', '^', '-', '\\', '/', '.', '$', '*', '+', '?', '(', ')', '{', '}', '|'}


class RangeType(StrEnum):
    POSITIVE = "positive"
    NEGATIVE = "negative"


class RegexExpr(ABC):
    """Base abstract class for all regex expressions"""
    @abstractmethod
    def __repr__(self) -> str:
        """Return the string representation of the regex expression"""
        pass

@dataclass
class CharacterRange(RegexExpr):
    """Represents a character range like [a-z] or [^0-9]"""
    from_char: str
    to_char: str
    range_type: RangeType = RangeType.POSITIVE

    def __post_init__(self):
        if self.from_char is not None and self.to_char is not None and self.from_char > self.to_char:
            raise ValueError(f"Invalid character range: '{self.from_char}' is greater than '{self.to_char}'")

    def __repr__(self):
        range_type = "^" if self.range_type == RangeType.NEGATIVE else ""
        
        # For single-character ranges, display without the range operator
        if self.from_char == self.to_char:
            # Escape special characters if needed
            char = f"\\{self.from_char}" if self.from_char in SPECIAL_CHARS else self.from_char
            if range_type:
                return f"[{range_type}{char}]"
            else:
                return f"{char}"
        
        # Escape special characters if needed
        from_char = f"\\{self.from_char}" if self.from_char in SPECIAL_CHARS else self.from_char
        to_char = f"\\{self.to_char}" if self.to_char in SPECIAL_CHARS else self.to_char
        
        return f"[{range_type}{from_char}-{to_char}]"

@dataclass
class CharacterSet(RegexExpr):
    """Represents a character set like [xyz] or [^aeiou]"""
    # TODO: it could also implement multiple ranges, e.g., [a-zA-Z0-9_]
    # FIXME: this assumes that chars and ranges are already in canonical form
    # e.g., no range overlaps and no duplicate characters, and CharacterRanges are always positive (but the whole CharSet can be negative)
    characters: Union[List[str | CharacterRange], set[str | CharacterRange]]
    range_type: RangeType = RangeType.POSITIVE

    def __post_init__(self):
        # Type-safe list comprehension that guarantees CharacterRange objects
        character_ranges: List[CharacterRange] = [
            c for c in self.characters 
            if isinstance(c, CharacterRange)
        ]
        assert all(c.range_type == RangeType.POSITIVE for c in character_ranges), \
            "CharacterSet can only contain positive CharacterRanges"
    
    def __repr__(self):
        range_type = "^" if self.range_type == RangeType.NEGATIVE else ""
        escaped_chars = []
        for char in self.characters:
            if isinstance(char, CharacterRange):
                # Strip the outer brackets from CharacterRange repr
                range_str = repr(char)
                escaped_chars.append(range_str[1:-1])  # Remove [ and ]
            elif char in SPECIAL_CHARS:
                escaped_chars.append('\\' + char)
            else:
                escaped_chars.append(char)
        return f"[{range_type}{''.join(escaped_chars)}]"

@dataclass
class Empty(RegexExpr):
    """Represents the empty language ∅"""
    def __repr__(self) -> str:
        return "∅"

@dataclass
class Epsilon(RegexExpr):
    """Represents the empty string ε"""
    def __repr__(self) -> str:
        return "ε"

@dataclass
class Concatenation(RegexExpr):
    """Represents a concatenation of regex expressions"""
    expressions: List[RegexExpr]

    def __post_init__(self):
        if len(self.expressions) < 1:
            raise ValueError("Concatenation must have at least one expression")

    def __repr__(self) -> str:
        # Add parentheses only if this is part of an alternation
        parts = []
        for expr in self.expressions:
            # If it's an alternation, we need its parentheses
            if isinstance(expr, Alternation):
                parts.append(str(expr))
            else:
                parts.append(str(expr))
        return ''.join(parts)

@dataclass
class Alternation(RegexExpr):
    """Represents a union (|) of regex expressions"""
    expressions: List[RegexExpr]

    def __post_init__(self):
        if len(self.expressions) < 1:
            raise ValueError("Alternation must have at least one expression")

    def __repr__(self) -> str:
        parts = []
        for expr in self.expressions:
            # For concatenations with multiple expressions, add parentheses
            if isinstance(expr, Concatenation) and len(expr.expressions) > 1:
                parts.append(f"({expr})")
            else:
                parts.append(str(expr))
        return f"({('|'.join(parts))})"

@dataclass
class KleeneStar(RegexExpr):
    """Represents a Kleene star (*) operation"""
    expr: RegexExpr

    def __repr__(self) -> str:
        # KleeneStar has higher precedence than alternation but lower than grouping
        # We need parentheses for alternation or concatenation with multiple elements
        needs_parens = (isinstance(self.expr, Alternation) or 
                       (isinstance(self.expr, Concatenation) and len(self.expr.expressions) > 1))
        base = str(self.expr)
        return f"({base})*" if needs_parens else f"{base}*"