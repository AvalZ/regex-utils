from abc import ABC, abstractmethod
from enum import StrEnum
from typing import Optional

# Characters that need to be escaped in regular expressions
SPECIAL_CHARS = {']', '[', '^', '-', '\\', '/', '.', '$', '*', '+', '?', '(', ')', '{', '}', '|'}

ALPHABET = set(chr(i) for i in range(256))


class RangeType(StrEnum):
    POSITIVE = "positive"
    NEGATIVE = "negative"


class RegexExpr(ABC):
    """Base abstract class for all regex expressions"""

    @abstractmethod
    def __contains__(self, char: str) -> bool:
        pass

    @abstractmethod
    def __repr__(self) -> str:
        pass

    @abstractmethod
    def derive(self, char: str) -> 'RegexExpr':
        pass

    @abstractmethod
    def is_nullable(self) -> bool:
        pass

    @abstractmethod
    def generate(self, custom_alphabet: Optional[set[str]] = None) -> Optional[str]:
        pass

    def normalize(self):
        return self

    def peek(self) -> 'RegexExpr':
        return self
