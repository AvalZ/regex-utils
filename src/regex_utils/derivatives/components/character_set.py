from enum import StrEnum
import random
from dataclasses import dataclass
from typing import Optional, Iterable

from regex_utils.derivatives.components.regex_expr import RegexExpr

# Characters that need to be escaped in regular expressions
SPECIAL_CHARS = {
    "]",
    "[",
    "^",
    "-",
    "\\",
    "/",
    ".",
    "$",
    "*",
    "+",
    "?",
    "(",
    ")",
    "{",
    "}",
    "|",
}

ALPHABET = set(chr(i) for i in range(256))


class RangeType(StrEnum):
    POSITIVE = "positive"
    NEGATIVE = "negative"

@dataclass
class CharacterSet(RegexExpr):
    """Represents a character set like [xyz] or [^aeiou]

    Note: the `characters` attribute is always coerced to a `set[str]`
    in __post_init__ so callers can pass a list or any iterable but the
    instance will store a set internally.
    """

    # Accept both lists and sets (and other iterables) at construction time;
    # __post_init__ will coerce to a set internally.
    characters: Iterable[str]
    range_type: RangeType = RangeType.POSITIVE

    def __post_init__(self) -> None:
        # Ensure characters is always a set internally. Accept lists/iterables.
        if not isinstance(self.characters, set):
            # cast any iterable to set
            try:
                self.characters = set(self.characters)
            except TypeError:
                # If characters is None or non-iterable, coerce to empty set
                self.characters = set()

    def __contains__(self, char: str):
        if self.range_type == RangeType.POSITIVE:
            return char in self.characters
        elif self.range_type == RangeType.NEGATIVE:
            return char not in self.characters
        else:
            raise ValueError("Invalid RangeType")

    def derive(self, char: str) -> "RegexExpr":
        from .empty import Empty
        from .epsilon import Epsilon

        return Epsilon() if char in self else Empty()

    def is_nullable(self) -> bool:
        return False

    def generate(self, custom_alphabet: Optional[set[str]] = None) -> Optional[str]:
        chars = set(self.characters)
        if self.range_type == RangeType.NEGATIVE:
            alphabet = custom_alphabet if custom_alphabet else ALPHABET
            valid_chars = list(alphabet - chars)
            return random.choice(sorted(valid_chars)) if valid_chars else None
        elif self.range_type == RangeType.POSITIVE:
            if not chars:
                return None
            return random.choice(sorted(list(chars)))
        else:
            raise ValueError("Invalid RangeType")

    def __repr__(self):
        # Coerce to a set for deterministic handling (len, sorting, etc.)
        chars = set(self.characters)
        # For deterministic output when characters is a set, sort by ordinal
        if len(chars) == 1 and self.range_type == RangeType.POSITIVE:
            return f"{next(iter(chars))}"

        range_type = "^" if self.range_type == RangeType.NEGATIVE else ""
        escaped_chars = []
        for char in sorted(chars, key=lambda c: ord(c)):
            if char in SPECIAL_CHARS:
                escaped_chars.append("\\" + char)
            else:
                escaped_chars.append(char)
        return f"[{range_type}{''.join(escaped_chars)}]"

    def __eq__(self, value: object) -> bool:
        """
        Equals if the range type is the same and the characters are the same
        """
        return (
            isinstance(value, CharacterSet)
            and self.range_type == value.range_type
            and set(self.characters) == set(value.characters)
        )

    def collapse(self) -> 'CharacterSet':
        return self
    
    def negate(self) -> 'CharacterSet':
        """Return a new CharacterSet that is the negation of this one."""
        new_range_type = (
            RangeType.NEGATIVE
            if self.range_type == RangeType.POSITIVE
            else RangeType.POSITIVE
        )
        return CharacterSet(self.characters, new_range_type)

    def get_explicit_charset(self, custom_alphabet: Optional[set[str]] = None) -> 'CharacterSet':
        """Return a new CharacterSet that explicitly lists all characters it matches.

        For positive character sets, this is the same as the original set.
        For negative character sets, this is the set of all characters in the
        alphabet (either custom_alphabet or full ASCII if none provided) that are not in the original set.
        """
        if self.range_type == RangeType.POSITIVE:
            return self

        alphabet = custom_alphabet if custom_alphabet else ALPHABET
        new_characters = alphabet - set(self.characters)

        return CharacterSet(
            new_characters,
        )