from abc import ABC, abstractmethod
from typing import Optional

class RegexExpr(ABC):
    """Base abstract class for all regex expressions"""

    @abstractmethod
    def __contains__(self, char: str) -> bool:
        pass

    @abstractmethod
    def __repr__(self) -> str:
        pass

    @abstractmethod
    def derive(self, char: str) -> "RegexExpr":
        pass

    @abstractmethod
    def is_nullable(self) -> bool:
        pass

    def generate(self, custom_alphabet: Optional[set[str]] = None) -> Optional[str]:
        return self.peek().collapse().generate(custom_alphabet)

    def normalize(self):
        return self

    def peek(self) -> "RegexExpr":
        return self

    @abstractmethod
    def collapse(self, custom_alphabet: set[str] | None = None) -> 'CharacterSet':
        pass