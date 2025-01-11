import random

from .transition import Direction, Transition
from .state import State, SynchronizedState
from regex_utils.utils import ALPHABET


class DFA:

    def __init__(self, alphabet=None, start_state=None):
        self.alphabet = alphabet or ALPHABET
        # Create an empty DFA
        if start_state:
            self.states = [start_state]
        else:
            self.states = [State()]
        self.start_state = self.states[0]
        self.transitions = []
        self.current_state = self.start_state

    def __contains__(self, item):
        if isinstance(item, State) or isinstance(item, SynchronizedState):
            return item in self.states
        elif isinstance(item, Transition):
            return item in self.transitions
        else:
            return False

    def get_state_by_id(self, id):
        return next((s for s in self.states if s.id == id), None)

    def get_end_states(self):
        return list(filter(lambda x: x.is_end_state, self.states))

    def reset(self):
        self.current_state = self.start_state

    def step(self):
        if self.current_state.get_transitions():
            t = random.choice(self.current_state.get_transitions())
            self.consume(t)
            return t.symbol
        else:
            return None

    def walk(self, continue_after_end_state=0.0):
        # TODO: add new walk techniques, e.g., weight walk on transitions that were already taken (to avoid being stuck in loops)
        self.reset()

        symbols = []
        while True:  # Otherwise it will stop at empty strings (epsilon-transitions)
            new_symbol = self.step()
            # raise error if stuck in dead end (i.e., new_symbol is None)
            # This usually means that the NFA does not exist, but it could also be a construction/simplification error
            if new_symbol is None:
                raise ValueError(
                    f"NFA is stuck in a dead end\nafter symbols: {symbols}"
                )

            symbols.append(new_symbol)
            if self.current_state.is_end_state:
                # Avoid generating a random number if continue_after_end_state is set to 0
                if (
                    continue_after_end_state == 0.0
                    or random.random() > continue_after_end_state
                    or not self.current_state.get_transitions()
                ):
                    break

        return "".join(symbols)

    def consume(self, transition):
        self.current_state = transition.to_state

    def append_transition(self, symbol):
        s = State()
        t = Transition(symbol, self.current_state, s)
        t.from_state.out_transitions.append(t)
        t.to_state.get_transitions(direction=Direction.BACKWARD).append(t)
        # Add transition and new state to the NFA
        self.transitions.append(t)
        self.states.append(s)
        self.current_state.is_end_state = False
        self.consume(t)
        return self

    def set_transition(
        self, symbol, from_state, to_state, direction=Direction.FORWARD, deep=True
    ):
        if direction == Direction.FORWARD:
            new_transition = Transition(symbol, from_state, to_state)
        elif direction == Direction.BACKWARD:
            new_transition = Transition(symbol, to_state, from_state)

        if new_transition not in self:
            self.transitions.append(new_transition)

        if deep:
            new_transition.get_next_state(Direction.BACKWARD).get_transitions(
                Direction.FORWARD
            ).append(new_transition)
            new_transition.get_next_state(Direction.FORWARD).get_transitions(
                Direction.BACKWARD
            ).append(new_transition)

        return new_transition

    def describe(self):
        for s in self.states:
            # Print state id and check if it's the start state
            print(f"State {s} is_end_state={s.is_end_state}")
            if s == self.start_state:
                print("\tStart state")
            for t in sorted(
                s.get_transitions(direction=Direction.BACKWARD), key=lambda x: x.symbol
            ):
                print(f"\t {t}")
            for t in sorted(s.get_transitions(), key=lambda x: x.symbol):
                print(f"\t\t -{t}")
    
    def get_equivalent_state(self, state):
        return next((s for s in self.states if s == state), None)
    
    def negate(self):
        """
        Negate the DFA
        """
        for state in self.states:
            state.is_end_state = not state.is_end_state

        return self
