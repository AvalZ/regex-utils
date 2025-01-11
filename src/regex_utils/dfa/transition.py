from enum import StrEnum


class Direction(StrEnum):
    FORWARD = "forward"
    BACKWARD = "backward"


class Transition:
    def __init__(self, symbol, from_state, to_state):
        self.symbol = symbol
        self.from_state = from_state
        self.to_state = to_state

    def is_self_loop(self):
        return self.from_state == self.to_state

    def is_epsilon_transition(self):
        return self.symbol == ""

    def get_next_state(self, direction=Direction.FORWARD):
        if direction == Direction.FORWARD:
            return self.to_state
        elif direction == Direction.BACKWARD:
            return self.from_state
        else:
            raise ValueError("Invalid direction")

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, Transition):
            return False
        return (
            self.symbol == value.symbol
            and self.from_state == value.from_state
            and self.to_state == value.to_state
        )

    def __hash__(self) -> int:
        return hash((self.symbol, self.from_state, self.to_state))

    def __str__(self) -> str:
        # use the "epsilon" greek letter instead of empty string
        symbol = "ε" if self.symbol == "" else self.symbol
        return f"({self.from_state})-{symbol}->({self.to_state})"

    def __repr__(self) -> str:
        return self.__str__()