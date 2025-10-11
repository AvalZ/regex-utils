from dataclasses import dataclass
import random
from typing import List, Optional, Union

from regex_utils.derivatives.components.regex_expr import RangeType, RegexExpr, ALPHABET, SPECIAL_CHARS


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
        from .empty import Empty
        from .epsilon import Epsilon
        return Epsilon() if char in self else Empty()

    def is_nullable(self) -> bool:
        return False

    def generate(self, custom_alphabet: Optional[set[str]] = None) -> Optional[str]:
        if self.range_type == RangeType.NEGATIVE:
            alphabet = custom_alphabet if custom_alphabet else ALPHABET
            valid_chars = list(alphabet - set(self.characters))
            return random.choice(valid_chars) if valid_chars else None
        elif self.range_type == RangeType.POSITIVE:
            if not self.characters:
                return None
            return random.choice(list(self.characters))
        else:
            raise ValueError("Invalid RangeType")

    def __repr__(self):
        range_type = "^" if self.range_type == RangeType.NEGATIVE else ""
        escaped_chars = []
        for char in self.characters:
            if char in SPECIAL_CHARS:
                escaped_chars.append('\\' + char)
            else:
                escaped_chars.append(char)
        return f"[{range_type}{''.join(escaped_chars)}]"
    
    def __eq__(self, value: object) -> bool:
        """
        Equals if the range type is the same and the characters are the same
        """
        return isinstance(value, CharacterSet) and self.range_type == value.range_type and set(self.characters) == set(value.characters)
