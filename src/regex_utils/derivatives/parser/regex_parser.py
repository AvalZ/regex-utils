
import sre_parse

from regex_utils.derivatives.components import RegexExpr, CharacterSet, Empty, Concatenation, KleeneStar, RangeType


class RegexParser:
    def __init__(self, regex_str: str):
        self.regex_str = regex_str

    def parse(self) -> RegexExpr:

        parsed = sre_parse.parse(self.regex_str)
        return self._parse_subpattern(parsed).normalize()

    def _parse_subpattern(self, subpattern: sre_parse.SubPattern) -> RegexExpr:

        if 0 == len(subpattern.data):
            raise ValueError("ERROR: regex is empty")
        elif 1 == len(subpattern.data):
            return self._sre_pattern_to_regex_expr(subpattern[0])
        else:
            elements = [self._sre_pattern_to_regex_expr(construct) for construct in subpattern.data]
            return Concatenation(elements)

    def _sre_pattern_to_regex_expr(self, construct: tuple | sre_parse.SubPattern) -> RegexExpr:
        # Implement the conversion logic from sre_parse constructs to RegexExpr
        type, value = construct

        if sre_parse.LITERAL == type: # a
            return CharacterSet([chr(value)])
        elif sre_parse.NOT_LITERAL == type:  # [^a]
            return CharacterSet([chr(value)], range_type=RangeType.NEGATIVE)
        elif sre_parse.IN == type:  # [abc] or [^abc]
            chars = []
            range_type = RangeType.POSITIVE
            for sub_type, sub_value in value:
                if sub_type == sre_parse.LITERAL:
                    chars.append(chr(sub_value))
                elif sub_type == sre_parse.NEGATE:
                    range_type = RangeType.NEGATIVE
                else:
                    raise NotImplementedError(f"Unsupported IN sub-construct: {sub_type}")
            return CharacterSet(chars, range_type=range_type)

        else:
            raise NotImplementedError(f"Unsupported construct: {type}")