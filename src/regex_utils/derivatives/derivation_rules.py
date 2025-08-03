from typing import Optional
from regex_utils.derivatives.parser import (
    RegexExpr,
    CharacterRange,
    CharacterSet,
    Concatenation,
    Alternation,
    KleeneStar,
    Empty,
    Epsilon,
)

"""
Derivation rules

a-1a	= ε
a-1b	= ∅	for each symbol b≠a
a-1ε	= ∅
a-1∅	= ∅
a-1(R*)	= (a-1R)R*
a-1(RS)	= (a-1R)S U v(R)a-1S
a-1(R & S)	= (a-1R) & (a-1S)
a-1(R U S)	= (a-1R) U (a-1S)
a-1(¬R)	= ¬(a-1R)
"""

class Derivative():
    def __init__(self, token: RegexExpr):
        self.token = token

def get_derivative_charset(regex: RegexExpr) -> CharacterRange | CharacterSet:
    """Get the derivative of a character set with respect to a character."""
    if isinstance(regex, CharacterRange):
        return CharacterRange(regex.from_char, regex.to_char, regex.range_type)
    elif isinstance(regex, CharacterSet):
        return CharacterSet(regex.characters, regex.range_type)
    raise ValueError(f"Unsupported regex type for derivative: {type(regex)}")

def derive(regex: RegexExpr, char: str) -> RegexExpr:
    """Compute the derivative of a regex with respect to a character.
    
    Args:
        regex: The regex expression to derive
        char: The character to derive with respect to
    
    Returns:
        The derivative of the regex, which is another regex expression
    """
    if isinstance(regex, CharacterRange) or isinstance(regex, CharacterSet):
        return derive_char(regex, char)
    elif isinstance(regex, Concatenation):
        return derive_concatenation(regex, char)
    elif isinstance(regex, Alternation):
        return derive_alternation(regex, char)
    elif isinstance(regex, KleeneStar):
        return derive_kleene_star(regex, char)
    elif isinstance(regex, Epsilon):
        return Empty()
    elif isinstance(regex, Empty):
        return Empty()
    raise ValueError(f"Unknown regex type: {type(regex)}")

def derive_char(regex: CharacterRange | CharacterSet, char: str) -> RegexExpr:
    """Derive a character range."""
    if isinstance(regex, CharacterRange):
        if ord(regex.from_char) >= ord(char) and ord(regex.to_char) <= ord(char):
            return Epsilon()
        return Empty()
    if isinstance(regex, CharacterSet):
        if char in regex.characters:
            return Epsilon()
        return Empty()

def derive_char_set(regex: CharacterSet, char: str) -> RegexExpr:
    """Derive a character set."""
    return Epsilon() if char in regex.characters else Empty()

def derive_concatenation(regex: Concatenation, char: str) -> RegexExpr:
    """Derive a concatenation of expressions.
    Rule: a-1(RS) = (a-1R)S U v(R)a-1S
    """
    if not regex.expressions:
        return Empty()
    
    if len(regex.expressions) == 1:
        return derive(regex.expressions[0], char)
    
    first_expr, second_expr, *remaining_expr = regex.expressions
        
    # FIXME: what if the second element is also nullable?
    if not is_nullable(first_expr):
        return Concatenation([derive(first_expr, char), second_expr] + remaining_expr)

    derived_expr = Alternation([
        Concatenation([derive(first_expr, char), second_expr]),
        derive(second_expr, char)
    ])

    return Concatenation([derived_expr] + remaining_expr) if remaining_expr else derived_expr


def derive_alternation(regex: Alternation, char: str) -> RegexExpr:
    """Derive an alternation of expressions.
    Rule: a-1(R|S) = a-1R | a-1S
    """
    if not regex.expressions:
        return Empty()
    
    derived_expressions = [derive(expr, char) for expr in regex.expressions]
    # Filter out Empty results
    non_empty_expressions = list(filter(lambda d: not isinstance(d, Empty), derived_expressions))
    
    if not non_empty_expressions:
        return Empty()
    elif len(non_empty_expressions) == 1:
        return non_empty_expressions[0]
    return Alternation(non_empty_expressions)

def derive_kleene_star(regex: KleeneStar, char: str) -> RegexExpr:
    """Derive a Kleene star expression.
    Rule: a-1(R*) = (a-1R)R*
    """
    d_expr = derive(regex.expr, char)
    if isinstance(d_expr, Empty):
        return Empty()
    return Concatenation([d_expr, regex])

def is_nullable(expr: RegexExpr) -> bool:
    """Check if a regex expression can match the empty string."""
    if isinstance(expr, (CharacterRange, CharacterSet, Empty)):
        return False
    elif isinstance(expr, Epsilon):
        return True
    elif isinstance(expr, Concatenation):
        return all(is_nullable(e) for e in expr.expressions)
    elif isinstance(expr, Alternation):
        return any(is_nullable(e) for e in expr.expressions)
    elif isinstance(expr, KleeneStar):
        return True
    raise ValueError(f"Unknown regex type: {type(expr)}")