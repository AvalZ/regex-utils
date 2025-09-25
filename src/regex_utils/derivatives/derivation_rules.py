from typing import Optional
from regex_utils.derivatives.parser import RegexExpr

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

def derive(regex: RegexExpr, char: str) -> RegexExpr:
    """Compute the derivative of a regex with respect to a character.
    
    Args:
        regex: The regex expression to derive
        char: The character to derive with respect to
    
    Returns:
        The derivative of the regex, which is another regex expression
    """
    return regex.derive(char).normalize()