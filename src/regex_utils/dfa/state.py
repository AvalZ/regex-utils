from .transition import Direction


class State:
    id_counter = 0

    def __init__(self, start_state=False, end_state=True, id=None):
        if id:
            self.id = id
        else:
            self.id = State.id_counter
            State.id_counter += 1
        self.out_transitions = []
        self.in_transitions = []
        self.is_start_state = start_state
        self.is_end_state = end_state

    def is_dead_end(self):
        return not self.is_end_state and (
            len(self.get_transitions()) == 0
            or all(t.is_self_loop() for t in self.get_transitions())
        )

    def is_dead_start(self):
        return not self.is_start_state and (
            not self.get_transitions(Direction.BACKWARD)
            or all(t.is_self_loop() for t in self.get_transitions(Direction.BACKWARD))
        )

    def get_outgoing_symbols(self, direction=Direction.FORWARD):
        return set(t.symbol for t in self.get_transitions(direction=direction))

    def get_transitions(self, direction=Direction.FORWARD, to_state=None, symbol=None):
        transitions = self._get_all_transitions_based_on_direction(direction)

        if to_state:
            transitions = list(filter(lambda x: x.to_state == to_state, transitions))
        if symbol is not None:
            transitions = list(filter(lambda x: x.symbol == symbol, transitions))

        return transitions
    
    def _get_all_transitions_based_on_direction(self, direction):
        transitions = []
        if direction == Direction.FORWARD:
            transitions = self.out_transitions
        elif direction == Direction.BACKWARD:
            transitions = self.in_transitions
        return transitions

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, State):
            return False
        return self.id == value.id

    def __hash__(self) -> int:
        return self.id

    def __str__(self) -> str:
        return f"{self.id}"

    def __repr__(self) -> str:
        return f"{self.id}"


class SynchronizedState(State):

    def __init__(self, *states, end_state=None, id=None):
        super().__init__(
            end_state=end_state or all(s.is_end_state for s in states), id=id
        )
        self.states = states

    def is_exit_state(self):
        return self.get_transitions(Direction.FORWARD) == []

    def is_entry_state(self):
        return self.get_transitions(Direction.BACKWARD) == []

    def __str__(self) -> str:
        return f"{self.id}:({",".join(str(s) for s in self.states)})"

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, SynchronizedState):
            return False
        return self.states == value.states

    def __hash__(self) -> int:
        return hash(self.states)


class StateSet(State):

    def __init__(self, *states, start_state=None, end_state=None, id=None):
        super().__init__(
            start_state=start_state,
            end_state=end_state,
            id=id
        )
        self.states = states

        if not self.is_end_state:
            self.is_end_state = any(s.is_end_state for s in self.states)
        if not self.is_start_state:
            self.is_start_state = any(s.is_start_state for s in self.states)

    def __str__(self) -> str:
        if self.states:
            return f"{self.id}:({",".join(str(s) for s in self.states)})"
        else:
            return f"{self.id}:Ø"

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, StateSet):
            return False
        return self.states == value.states

    def __hash__(self) -> int:
        return hash(frozenset(self.states))
    
    def is_dead_state(self):
        return self.states == set()
    
    def get_inner_transitions(self, direction=Direction.FORWARD, symbol=None):
        transitions = []
        for state in self.states:
            transitions.extend(state.get_transitions(direction=direction, symbol=symbol))
        return transitions

    def get_inner_states(self):
        return self.states
