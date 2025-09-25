from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from typing import List, Union

# Characters that need to be escaped in regular expressions
SPECIAL_CHARS = {']', '[', '^', '-', '\\', '/', '.', '$', '*', '+', '?', '(', ')', '{', '}', '|'}


class RangeType(StrEnum):
    POSITIVE = "positive"
    NEGATIVE = "negative"


class RegexExpr(ABC):
    """Base abstract class for all regex expressions"""

    @abstractmethod
    def __contains__(self, char: str) -> bool:
        """Check if the character is contained in the regex expression"""
        pass

    @abstractmethod
    def __repr__(self) -> str:
        """Return the string representation of the regex expression"""
        pass

    @abstractmethod
    def derive(self, char: str) -> 'RegexExpr':
        """Compute the derivative of the regex with respect to a character"""
        pass

    @abstractmethod
    def is_nullable(self) -> bool:
        """Check if the regex expression can match the empty string"""
        pass

    def normalize(self):
        return self

@dataclass
class CharacterSet(RegexExpr):
    """Represents a character set like [xyz] or [^aeiou]"""
    characters: Union[List[str], set[str]]
    range_type: RangeType = RangeType.POSITIVE


    def __contains__(self, char: str):
        if self.range_type == RangeType.POSITIVE:
            return char in self.characters
        elif self.range_type == RangeType.NEGATIVE:
            return char not in self.characters
        else:
            raise ValueError("Invalid RangeType")

    def derive(self, char: str) -> 'RegexExpr':
        return Epsilon() if char in self else Empty()

    def is_nullable(self) -> bool:
        return False

    def __repr__(self):
        range_type = "^" if self.range_type == RangeType.NEGATIVE else ""
        escaped_chars = []
        for char in self.characters:
            if char in SPECIAL_CHARS:
                escaped_chars.append('\\' + char)
            else:
                escaped_chars.append(char)
        return f"[{range_type}{''.join(escaped_chars)}]"

@dataclass
class Empty(RegexExpr):
    """Represents the empty language ∅"""


    def __contains__(self, _: str) -> bool:
        # Empty does not contain any characters
        return False

    def derive(self, char: str) -> 'RegexExpr':
        return self

    def is_nullable(self) -> bool:
        return False

    def __repr__(self) -> str:
        return "∅"

@dataclass
class Epsilon(RegexExpr):
    """Represents the empty string ε"""


    def __contains__(self, char: str) -> bool:
        # Epsilon only contains the empty string
        return '' == char

    def derive(self, char: str) -> 'RegexExpr':
        return Empty()

    def is_nullable(self) -> bool:
        return True

    def __repr__(self) -> str:
        return "ε"

@dataclass
class Concatenation(RegexExpr):
    """Represents a concatenation of regex expressions"""
    expressions: List[RegexExpr]


    def __contains__(self, char: str) -> bool:
        return char in self.expressions[0]

    def derive(self, char: str) -> 'RegexExpr':
        if not self.expressions:
            return Empty()
        # TODO: add a normalization function that removes Concatenations with 1 expression only
        if len(self.expressions) == 1:
            return self.expressions[0].derive(char)

        first_expr, *remaining_expr = self.expressions

        if not first_expr.is_nullable():
            return Concatenation([first_expr.derive(char), *remaining_expr] + remaining_expr)

        derived_expr = Alternation([
            Concatenation([first_expr.derive(char), *remaining_expr]),
            Concatenation(remaining_expr).derive(char)
        ])
        return Concatenation([derived_expr] + remaining_expr) if remaining_expr else derived_expr

    def is_nullable(self) -> bool:
        return all(expr.is_nullable() for expr in self.expressions)
    
    def normalize(self) -> 'RegexExpr':
        # If there's only one expression, return it directly
        if len(self.expressions) == 1:
            return self.expressions[0].normalize()

        if not self.expressions:
            return Epsilon()

        first_expression = self.expressions[0].normalize()

        if first_expression and isinstance(first_expression, Empty):
            return Empty()
        
        if first_expression and isinstance(first_expression, Epsilon):
            return Concatenation(self.expressions[1:]).normalize()
        
        return Concatenation([first_expression, *self.expressions[1:]])

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


    def __contains__(self, char: str) -> bool:
        return any(char in expr for expr in self.expressions)

    def derive(self, char: str) -> 'RegexExpr':
        derived_expressions = [expr.derive(char) for expr in self.expressions]
        non_empty_expressions = [d for d in derived_expressions if not isinstance(d, Empty)]
        if not non_empty_expressions:
            return Empty()
        elif len(non_empty_expressions) == 1:
            return non_empty_expressions[0]
        return Alternation(non_empty_expressions)

    def is_nullable(self) -> bool:
        return any(expr.is_nullable() for expr in self.expressions)

    def normalize(self) -> 'RegexExpr':
        if all(isinstance(expr, Epsilon) for expr in self.expressions):
            return Epsilon()

        # Remove Empty and Epsilon expressions
        self.expressions = [expr for expr in self.expressions if not isinstance(expr, (Empty, Epsilon))]

        if not self.expressions:
            return Empty()
        
        if len(self.expressions) == 1:
            return self.expressions[0].normalize()
        
        normalized_expressions = self.expressions[:]
        while any(isinstance(expr, Alternation) for expr in normalized_expressions):
            normalized_expressions = []

            for expr in self.expressions:
                expr.normalize()
            
                if isinstance(expr, Alternation):
                    normalized_expressions.extend([e.normalize() for e in expr.expressions])
                else:
                    normalized_expressions.append(expr)

            if all(isinstance(expr, Empty) for expr in normalized_expressions):
                return Empty()
            
            if all(isinstance(expr, Epsilon) for expr in normalized_expressions):
                return Epsilon()
            
            # Remove Empty and Epsilon expressions
            normalized_expressions = [expr for expr in normalized_expressions if not isinstance(expr, (Empty, Epsilon))]

            self.expressions = normalized_expressions

        return Alternation(normalized_expressions)

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
    expression: RegexExpr


    def __contains__(self, char: str) -> bool:
        # A Kleene star can generate the empty string, ma non un carattere specifico
        return char in self.expression

    def derive(self, char: str) -> 'RegexExpr':
        d_expr = self.expression.derive(char)
        if isinstance(d_expr, Empty):
            return Empty()
        return Concatenation([d_expr, self])

    def is_nullable(self) -> bool:
        return True

    def __repr__(self) -> str:
        # KleeneStar has higher precedence than alternation but lower than grouping
        # We need parentheses for alternation or concatenation with multiple elements
        needs_parens = (isinstance(self.expression, Alternation) or 
                       (isinstance(self.expression, Concatenation) and len(self.expression.expressions) > 1))
        base = str(self.expression)
        return f"({base})*" if needs_parens else f"{base}*"