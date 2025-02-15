import more_itertools

import re

ALPHABET = set([chr(i) for i in range(0, 256)])

WORDCHARACTERS = set([chr(i) for i in range(48, 58)]) | set([chr(i) for i in range(65, 91)]) | set([chr(i) for i in range(97, 123)]) | set(["_"])
NONWORDCHARACTERS = ALPHABET - WORDCHARACTERS
SPACECHARACTERS = set([chr(i) for i in range(9, 14)] + [" "])


def range_label(symbols: set, alphabet=None):
    if alphabet is None:
        alphabet = ALPHABET
    if not symbols:
        return ""

    # Create compact label
    if len(symbols) == len(alphabet):
        label = "."  # All symbols
    elif symbols == set(range(0, 10)):
        label = "\\d"
    elif symbols == SPACECHARACTERS:
        label = "\\s"
    elif symbols == WORDCHARACTERS:
        label = "\\w"
    elif symbols == NONWORDCHARACTERS:
        label = "\\W"
    elif symbols and len(symbols) > 1:
        symbols = sorted(list(symbols))
        if len(symbols) < len(alphabet) / 2:
            label = "["
        else:
            label = "[^"
            symbols = list(sorted(alphabet - set(symbols)))
        # concatenate only if the symbols are close to each other
        for r in more_itertools.consecutive_groups(symbols, ordering=lambda x: ord(x)):
            r = list(r)
            if len(r) > 3:
                label += f"{escape_symbol(r[0])}-{escape_symbol(r[-1])}"
            else:
                label += "".join([escape_symbol(s) for s in r])
        label += "]"
    else:
        label = escape_symbol(symbols.pop())

    return label


def escape_symbol(symbol):
    # Escape if \r, \n, \t, \f, \v
    if ord(symbol) < 32:
        return repr(symbol)[1:-1]

    return re.escape(symbol)
