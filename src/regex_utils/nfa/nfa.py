import copy
import random
import sre_constants
import sre_parse
from enum import StrEnum

import graphviz
from graphviz.quoting import *

from regex_utils.nfa import utils
from collections import deque


class Direction(StrEnum):
    FORWARD = "forward"
    BACKWARD = "backward"


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

    def get_all_outgoing_symbols(self):
        symbols = set(t.symbol for t in self.get_transitions())

        states = {self}

        while True:
            previous_states = states.copy()

            for t in self.get_transitions():
                if t.is_epsilon_transition() and not t.is_self_loop():
                    symbols.update(t.to_state.get_all_outgoing_symbols())
                    states.add(t.to_state)

            if states == previous_states:
                break

        return symbols

    def complement_outbound_transitions(self):
        return set(self.alphabet) - self.get_all_outgoing_symbols()

    def get_transitions(self, direction=Direction.FORWARD, to_state=None, symbol=None):
        transitions = self._get_all_transitions_based_on_direction(direction)

        if to_state:
            transitions = list(filter(lambda x: x.to_state == to_state, transitions))
        if symbol:
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

    def __init__(self, s1, s2, end_state=None, id=None):
        super().__init__(
            end_state=end_state or (s1.is_end_state and s2.is_end_state), id=id
        )
        self.states = (s1, s2)

    def is_exit_state(self):
        return self.get_transitions(Direction.FORWARD) == []
    
    def is_entry_state(self):
        return self.get_transitions(Direction.BACKWARD) == []

    def __str__(self) -> str:
        return f"{self.id}:({self.states[0]},{self.states[1]})"

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, SynchronizedState):
            return False
        return self.states == value.states

    def __hash__(self) -> int:
        return hash(self.states)


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


ENUM_LOOKAROUND_TYPE = {
    sre_constants.ASSERT: "POSITIVE",
    sre_constants.ASSERT_NOT: "NEGATIVE",
}

ENUM_LOOKAROUND_DIRECTION = {-1: "BEHIND", 1: "AHEAD"}


class Lookaround:
    def __init__(
        self, from_state, lookaround_type, lookaround_direction, lookaround_pattern
    ):
        self.from_state = from_state
        self.lookaround_type = lookaround_type
        self.lookaround_direction = lookaround_direction
        self.lookaround_pattern = lookaround_pattern


class Boundary:
    def __init__(self, from_state, boundary_type):
        self.from_state = from_state
        self.boundary_type = boundary_type
    
    def __repr__(self) -> str:
        return f"Boundary({self.from_state}, {self.boundary_type})"
    
    def __str__(self) -> str:
        return self.__repr__()


class NFA:
    def __init__(self, alphabet=None):
        self.alphabet = alphabet or utils.ALPHABET
        # Create an empty NFA
        self.boundaries = []
        self.states = [State()]
        self.transitions = []
        self.start_state = self.states[0]
        self.current_state = self.start_state
        # Track lookarounds
        self.lookarounds = []
    
    def is_empty(self):
        if not self.transitions or all(t.is_epsilon_transition() for t in self.transitions):
            return True
        return False
    
    def __bool__(self):
        return not self.is_empty()

    def __contains__(self, item):
        if isinstance(item, State) or isinstance(item, SynchronizedState):
            return item in self.states
        elif isinstance(item, Transition):
            return item in self.transitions
        elif isinstance(item, Lookaround):
            return item in self.lookarounds
        elif isinstance(item, Boundary):
            return item in self.boundaries
        else:
            return False

    def _add_merged_state_to_merged_nfa(
        self, s1, s2, transition_symbol, current_sync_state, direction=Direction.FORWARD
    ):
        new_state_added = False
        new_sync_state = SynchronizedState(s1, s2)
        # print(f"Adding new state {new_sync_state}")

        if new_sync_state not in self:
            # print(f"State {new_sync_state} not in NFA")
            # print(f"self.states: {self.states}")
            new_state_added = True
            self.states.append(new_sync_state)

        if not new_state_added:
            new_sync_state = next(
                filter(lambda x: x == new_sync_state, self.states), None
            )

        self.set_transition(
            transition_symbol, current_sync_state, new_sync_state, direction=direction
        )

        return new_state_added, new_sync_state

    def merge(
        self,
        nfa2,
        direction=Direction.FORWARD,
        nfa1_sync_state=None,
        nfa2_sync_state=None,
    ):
        """
        Intersect two NFAs by creating a new NFA that accepts strings accepted by both input NFAs
        Basically an AND for NFAs

        :param nfa1:
        :param nfa2:
        :param nfa1_sync_state: optional state in nfa1 to sync on
        :param nfa2_sync_state: optional state in nfa2 to sync on
        :return:
        """

        nfa1 = self.simplify()
        nfa2.simplify()

        # TODO: move this into a SynchronizedNFA class
        merged_nfa = NFA()
        merged_nfa.start_state = None
        merged_nfa.states = []

        if nfa1_sync_state is None:
            if direction == Direction.FORWARD:
                nfa1_sync_state = nfa1.start_state
            if direction == Direction.BACKWARD:
                nfa1_sync_state = nfa1._fix().get_end_states()[0]

        if nfa2_sync_state is None:
            if direction == Direction.FORWARD:
                nfa2_sync_state = nfa2.start_state
            if direction == Direction.BACKWARD:
                nfa2_sync_state = nfa2._fix().get_end_states()[0]

        starting_synchronized_state = SynchronizedState(
            nfa1_sync_state, nfa2_sync_state
        )

        merged_nfa.states.append(starting_synchronized_state)

        # FIXME: this is a workaround to fix the start state of the merged NFA
        # but we're not currently using start_state in SynchronizedStates
        merged_nfa.start_state = starting_synchronized_state

        states_frontier = deque([starting_synchronized_state])

        while states_frontier:

            sync_state = states_frontier.popleft()

            # get all transitions from states in the synchronized states
            transitions1 = nfa1.get_transitions_from(sync_state.states[0], direction)
            transitions2 = nfa2.get_transitions_from(sync_state.states[1], direction)

            epsilon_transitions1 = list(
                filter(lambda t: t.is_epsilon_transition(), transitions1)
            )
            epsilon_transitions2 = list(
                filter(lambda t: t.is_epsilon_transition(), transitions2)
            )

            if epsilon_transitions1 and epsilon_transitions2:
                for t1 in epsilon_transitions1:
                    for t2 in epsilon_transitions2:
                        new_state_added, new_sync_state = (
                            merged_nfa._add_merged_state_to_merged_nfa(
                                t1.get_next_state(direction),
                                t2.get_next_state(direction),
                                "",
                                sync_state,
                                direction=direction,
                            )
                        )

                        if new_state_added:
                            states_frontier.append(new_sync_state)


            else:
                # Consume any epsilon transitions in nfa1 and nfa2
                # and create new synchronized states for each, if they don't exist already
                for t1 in list(filter(lambda t: t.is_epsilon_transition(), transitions1)):
                    next_s1_state = t1.get_next_state(direction)

                    new_state_added, new_sync_state = (
                        merged_nfa._add_merged_state_to_merged_nfa(
                            next_s1_state,
                            sync_state.states[1],
                            "",
                            sync_state,
                            direction=direction,
                        )
                    )

                    if new_state_added:
                        states_frontier.append(new_sync_state)

                for t2 in filter(lambda t: t.is_epsilon_transition(), transitions2):
                    next_s2_state = t2.get_next_state(direction)

                    new_state_added, new_sync_state = (
                        merged_nfa._add_merged_state_to_merged_nfa(
                            sync_state.states[0],
                            next_s2_state,
                            "",
                            sync_state,
                            direction=direction,
                        )
                    )

                    if new_state_added:
                        states_frontier.append(new_sync_state)

            # For each transition from s1, check if there's a corresponding transition from s2
            # There could also be multiple transitions from s2 with the same symbol, in which case we need to create a new synchronized state for each
            for t1 in filter(lambda t: not t.is_epsilon_transition(), transitions1):
                matching_transitions = filter(
                    lambda t2: t1.symbol == t2.symbol
                    and not t2.is_epsilon_transition(),
                    transitions2,
                )
                for t2 in matching_transitions:
                    next_s1_state = t1.get_next_state(direction)
                    next_s2_state = t2.get_next_state(direction)

                    new_state_added, new_sync_state = (
                        merged_nfa._add_merged_state_to_merged_nfa(
                            next_s1_state,
                            next_s2_state,
                            t1.symbol,
                            sync_state,
                            direction=direction,
                        )
                    )

                    if new_state_added:
                        states_frontier.append(new_sync_state)

        # propagate boundaries to synchronized states
        for b in nfa1.boundaries + nfa2.boundaries:
            for sync_state in merged_nfa.states:
                if (
                    b.from_state is sync_state.states[0]
                    or b.from_state is sync_state.states[1]
                ):
                    merged_nfa.boundaries.append(Boundary(sync_state, b.boundary_type))
                    # TODO check if boundaries are transferred to the original NFA

        return merged_nfa

    def merge_boundaries(self):
        for b in list(self.boundaries):
            if b.from_state not in self.states:
                break
            # Word boundaries
            if b.boundary_type == sre_constants.AT_BOUNDARY:
                self._merge_word_boundary(b)

        return self

    def _epsilon_closure(self, state) -> set:
        closure = {state}
        while True:
            previous_closure = closure.copy()

            for t in state.get_transitions():
                if t.is_epsilon_transition():
                    closure.add(t.to_state)

            for t in state.get_transitions(direction=Direction.BACKWARD):
                if t.is_epsilon_transition():
                    closure.add(t.from_state)

            if closure == previous_closure:
                break

        return closure

    def get_transitions_from(self, state_or_closure, direction=Direction.FORWARD):
        if isinstance(state_or_closure, State) or isinstance(
            state_or_closure, SynchronizedState
        ):
            return state_or_closure.get_transitions(direction=direction)
        else:
            transitions = []
            for s in state_or_closure:
                transitions.extend(s.get_transitions(direction=direction))
            return transitions

    def concatenate(self, other_nfa, on_end_states=None, on_other_nfa_start_states=None):
        if on_end_states is None:
            end_states = self.get_end_states()
        else:
            end_states = on_end_states

        if on_other_nfa_start_states is None:
            other_nfa_start_states = [other_nfa.start_state]
        else:
            other_nfa_start_states = on_other_nfa_start_states

        for end_state in end_states:
            for start_state in other_nfa_start_states:
                self.set_epsilon_transition(end_state, start_state)
                end_state.is_end_state = False

        self.states.extend(other_nfa.states)
        self.transitions.extend(other_nfa.transitions)
        self.lookarounds.extend(other_nfa.lookarounds)
        self.boundaries.extend(other_nfa.boundaries)
        return self

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

    def set_transition(self, symbol, from_state, to_state, direction=Direction.FORWARD, deep=True):
        if direction == Direction.FORWARD:
            new_transition = Transition(symbol, from_state, to_state)
        elif direction == Direction.BACKWARD:
            new_transition = Transition(symbol, to_state, from_state)

        if new_transition not in self:
            self.transitions.append(new_transition)

        if deep:
            new_transition.get_next_state(Direction.BACKWARD).get_transitions(Direction.FORWARD).append(new_transition)
            new_transition.get_next_state(Direction.FORWARD).get_transitions(Direction.BACKWARD).append(new_transition)

        return new_transition

    def set_epsilon_transition(self, from_state, to_state):
        return self.set_transition("", from_state, to_state)

    def make_skippable(self):
        for s in filter(lambda x: x.is_end_state, self.states):
            self.set_epsilon_transition(self.start_state, s)
        return self

    def make_kleene(self):
        self.make_skippable()
        for s in filter(lambda x: x.is_end_state, self.states):
            self.set_epsilon_transition(s, self.start_state)
        return self

    def force_merge_states(self, merged_state, removed_state):
        """
        Merge two states in the NFA by transferring all transitions from removed_state to merged_state
        WARNING: this may create unwanted self-loops (removed in the simplify function)

        :param merged_state:
        :param removed_state:
        :return:
        """
        if merged_state is removed_state:
            return self
        if merged_state not in self.states or removed_state not in self.states:
            return self
        # Transfer all transitions from removed_state to merged_state
        # WARNING: this may create unwanted self-loops
        for t in removed_state.get_transitions():
            t.from_state = merged_state
            merged_state.get_transitions().append(t)

        for t in removed_state.get_transitions(direction=Direction.BACKWARD):
            t.to_state = merged_state
            merged_state.get_transitions(direction=Direction.BACKWARD).append(t)

        if removed_state.is_end_state:
            merged_state.is_end_state = True

        if removed_state is self.start_state:
            self.start_state = merged_state

        # Update lookaheads and boundaries on removed state

        lookaheads_on_removed_state = list(
            filter(lambda x: x.from_state == removed_state, self.lookarounds)
        )
        boundaries_on_removed_state = list(
            filter(lambda x: x.from_state == removed_state, self.boundaries)
        )

        for lookahead in lookaheads_on_removed_state:
            lookahead.from_state = merged_state

        for b in boundaries_on_removed_state:
            b.from_state = merged_state

        try:
            self.states.remove(removed_state)
        except ValueError:
            print("Cannot remove state", removed_state)

        return self

    def get_state_by_id(self, id):
        return next((s for s in self.states if s.id == id), None)

    def simplify(self, budget=1000):
        """
        Simplify the NFA by using multiple simplification techniques
        Stops on a budget or when simplifications stop changing the NFA (fixed point)

        :param budget:
        :return:
        """

        rounds = 0
        while True:

            # If start states has no outgoing transitions, the NFA is empty
            if not self.start_state.get_transitions():
                # remove all other states and transitions
                for t in list(self.transitions):
                    self.remove_transition(t)

                for s in list(self.states):
                    if s is not self.start_state:
                        self.states.remove(s)
                return self

            previous_states = len(self.states)
            previous_transitions = len(self.transitions)
            previous_end_states = len(
                list(filter(lambda x: x.is_end_state, self.states))
            )

            self.remove_useless_epsilon_transitions()
            self.remove_two_way_epsilon_transitions()
            self.remove_epsilon_self_loops()

            self.remove_dead_end_states()
            self._remove_dead_start_states()
            self.remove_duplicate_transitions()
            self.remove_dead_end_transitions()

            self.propagate_end_states()

            rounds += 1
            if rounds > budget or (
                previous_states == len(self.states)
                and previous_transitions == len(self.transitions)
                and previous_end_states
                == len(list(filter(lambda x: x.is_end_state, self.states)))
            ):
                if rounds > budget:
                    print("[!] Simplify reached budget")
                break

        # self._rewrite_states_ids()
        return self

    def _remove_dead_start_states(self):
        # FIXME use is_dead_start function from State class
        # Remove all states that have no in_transitions and are not the start_state
        dead_start_states = [
            s
            for s in self.states
            if not s.get_transitions(direction=Direction.BACKWARD)
            and s is not self.start_state
        ]

        # Remove all transitions from dead start states
        for s in dead_start_states:
            for t in list(s.get_transitions()):
                self.remove_transition(t)
            self.states.remove(s)

        return self

    def propagate_end_states(self):
        # Propagate end states
        # If there is an epsilon transition from a state to an end state,
        # mark the state as an end state
        for s in self.states:
            if any(
                filter(
                    lambda x: x.is_epsilon_transition() and x.to_state.is_end_state,
                    s.get_transitions(),
                )
            ):
                s.is_end_state = True

    def remove_dead_end_transitions(self):
        # Dead transition = a transition that
        # Has no from_state OR has no to_state
        dead_transitions = [
            t
            for t in self.transitions
            if t.from_state not in self.states or t.to_state not in self.states
        ]
        for t in dead_transitions:
            self.remove_transition(t)

    def remove_two_way_epsilon_transitions(self):
        # Remove two-way epsilon transitions
        # - Find all two-way epsilon transitions between two states
        to_merge_states = []
        for s in self.states:
            for t in s.get_transitions():
                if t.is_epsilon_transition() and not t.is_self_loop():
                    for t2 in t.to_state.get_transitions():
                        if (
                            t2.to_state == s
                            and t2.is_epsilon_transition()
                            and not t2.is_self_loop()
                        ):
                            to_merge_states.append((t.from_state, t.to_state))

        for from_state, to_state in to_merge_states:
            self.force_merge_states(from_state, to_state)

        return self

    def remove_useless_epsilon_transitions(self):
        for t in self.transitions:
            if t.symbol == "":
                # If there are no other transitions between from_state and to_state
                if (
                    (
                        len(t.from_state.get_transitions()) == 1
                        or len(t.to_state.get_transitions(direction=Direction.BACKWARD))
                        == 1
                    )
                    # the state is not the start of a boundary
                    and not any(
                        filter(
                            lambda x: x.from_state in [t.to_state, t.from_state],
                            self.boundaries + self.lookarounds,
                        )
                    )
                ):
                    self.force_merge_states(t.from_state, t.to_state)

        return self

    def remove_epsilon_self_loops(self):
        # Remove epsilon self-loops
        for t in self.transitions:
            if t.is_epsilon_transition() and t.is_self_loop():
                self.remove_transition(t)
        return self

    def remove_dead_end_states(self):
        # Dead state = a state that
        # Is not a final state, and
        # Has no outgoing transitions OR has all outgoing transitions as self-loops

        while True:
            previous_states = len(self.states)
            dead_end_states = [s for s in self.states if s.is_dead_end()]
            for s in dead_end_states:
                # Remove dead end state
                self.states.remove(s)
                # And all transitions to it
                for t in s.get_transitions(direction=Direction.BACKWARD):
                    self.remove_transition(t)

            # Exit if it didn't remove any states, otherwise repeat
            # since removing transitions could have created new dead-end states
            if previous_states == len(self.states):
                return self

    def remove_duplicate_transitions(self):
        # Remove duplicate transitions
        for s in self.states:
            # Group transitions by symbol and to_state
            for t in s.get_transitions():
                duplicate_transitions = [
                    t2
                    for t2 in s.get_transitions()
                    if t2.symbol == t.symbol and t2.to_state == t.to_state
                ]
                for dt in duplicate_transitions[1:]:
                    self.remove_transition(dt)

    def negate_range_transition_between(self, from_state, to_state):
        # There must be a transition between the two states
        assert any(
            filter(
                lambda x: x.from_state == from_state and x.to_state == to_state,
                self.transitions,
            )
        )
        # All transitions must be 1 char only, or epsilon transitions
        assert all(
            [
                len(t.symbol) == 1 or t.is_epsilon_transition()
                for t in filter(
                    lambda x: x.from_state == from_state and x.to_state == to_state,
                    self.transitions,
                )
            ]
        )

        # Get all transitions between the two states
        transitions = list(
            filter(
                lambda x: x.from_state == from_state and x.to_state == to_state,
                self.transitions,
            )
        )
        # Get all symbols in the transitions
        symbols = [t.symbol for t in transitions]
        # Get all symbols not in the transitions
        not_symbols = list(set(self.alphabet) - set(symbols))

        # FIXME: Regex don't have this behavior except when at end of string (sometimes)
        # if '' not in symbols:
        #     not_symbols.append('')

        # Create negated transitions
        for s in not_symbols:
            self.set_transition(s, from_state, to_state)

        # Remove all transitions between the two states
        for t in transitions:
            self.remove_transition(t)

        return self

    def remove_transition(self, transition):
        try:
            self.transitions.remove(transition)
        except ValueError:
            pass

        try:
            transition.from_state.get_transitions().remove(transition)
        except ValueError:
            pass

        try:
            transition.to_state.get_transitions(direction=Direction.BACKWARD).remove(
                transition
            )
        except ValueError:
            pass

    def remove_state(self, state):
        for t in state.get_transitions():
            self.remove_transition(t)
        for t in state.get_transitions(direction=Direction.BACKWARD):
            self.remove_transition(t)
        
        for b in self.boundaries:
            if b.from_state == state:
                self.boundaries.remove(b)

        self.states.remove(state)

    def to_dot(self, view=True, simplified=False):
        """
        Generate a DOT representation of the NFA
        End states are marked as double circles

        :param view:
        :param simplified:
        :return:
        """

        if simplified:
            self.simplify()

        dot = graphviz.Digraph()
        dot.attr(rankdir="LR")
        for s in self.states:
            dot.node(
                str(s.id),
                # Make the circle double if endstate
                shape="doublecircle" if s.is_end_state else "circle",
                # make the circle red if there's a lookaround on this state
                color=(
                    "red"
                    if any(
                        filter(
                            lambda x: x.from_state == s,
                            self.lookarounds + self.boundaries,
                        )
                    )
                    else "black"
                ),
                label=str(s),
            )

        # Group transitions by from_state and to_state
        # If from_state and to_state are the same for multiple transitions,
        # group in a single edge and show a range instead as a label
        excluded_transitions = []

        for t in self.transitions:
            if t in excluded_transitions:
                continue
            symbols = set()
            same_transitions = self._get_parallel_transitions(
                t, get_epsilon_transitions=True
            )
            if same_transitions:
                symbols.update(t.symbol for t in same_transitions)

            if "" in symbols:
                dot.edge(str(t.from_state.id), str(t.to_state.id), label="ε")
                symbols = symbols - {""}

            if t.symbol:
                symbols.add(t.symbol)

            label = utils.range_label(symbols)

            if label:
                # ("Creating transition", t.from_state.id, t.to_state.id, repr(label))
                dot.edge(
                    str(t.from_state.id),
                    str(t.to_state.id),
                    label=escape(repr(label))[1:-1],
                )

            for st in same_transitions:
                excluded_transitions.append(st)

        # Workaround to mark start state
        dot.node("", shape="none", width="0")
        dot.edge("", str(self.start_state.id))

        dot.render(view=view)

        return dot.source

    def _rewrite_states_ids(self):
        # This is technically not needed, but adding it due to some simplification mishaps
        all_states = set(
            self.states
            + [t.from_state for t in self.transitions]
            + [t.to_state for t in self.transitions]
        )

        for i, s in enumerate(all_states):
            s.id = i

        return self

    def describe(self):
        for s in self.states:
            # Print state id and check if it's the start state
            print(f"State {s.id} is_end_state={s.is_end_state}")
            if s == self.start_state:
                print("\tStart state")
            for t in sorted(
                s.get_transitions(direction=Direction.BACKWARD), key=lambda x: x.symbol
            ):
                print(f"\t {t.from_state.id} -{t.symbol}->")
            for t in sorted(s.get_transitions(), key=lambda x: x.symbol):
                print(f"\t\t -{t.symbol}-> {t.to_state.id}")

    def _fix(self):
        """
        Add a fake start state and end state to the NFA

        Mainly used in the to_regex function for the state elimination algorithm

        :return:
        """
        start_state = State(end_state=False)
        end_state = State()
        self.states.append(start_state)
        self.set_epsilon_transition(start_state, self.start_state)
        for s in self.get_end_states():
            self.set_epsilon_transition(s, end_state)
            s.is_end_state = False

        self.states.append(end_state)
        self.start_state = start_state
        return self

    def _get_transitions_between_states(
        self, from_state, to_state, get_epsilon_transitions=False
    ):
        transitions = list(
            filter(
                lambda x: x.from_state == from_state and x.to_state == to_state,
                self.transitions,
            )
        )
        if not get_epsilon_transitions:
            transitions = list(
                filter(lambda x: not x.is_epsilon_transition(), transitions)
            )

        return transitions

    def _get_parallel_transitions(self, transition, get_epsilon_transitions=False):
        return list(
            self._get_transitions_between_states(
                transition.from_state, transition.to_state, get_epsilon_transitions
            )
        )

    def _merge_parallel_transitions(self):
        """
        Merge parallel transitions between states in the NFA
        WARNING: as of now, this is disruptive for the NFA
        it will break the `walk` function (and others) if called after it
        :return:
        """
        # Merge parallel transitions
        for s in self.states:
            for s2 in [t.to_state for t in s.get_transitions()]:
                transitions = self._get_transitions_between_states(
                    s, s2, get_epsilon_transitions=True
                )

                if transitions:
                    for t in transitions:
                        self.remove_transition(t)
                    # Create a new transition
                    symbols = set(t.symbol for t in transitions)
                    if all(len(symbol) == 1 or not symbol for symbol in symbols):
                        label = utils.range_label(symbols - {""})
                    else:
                        # FIXME this gives wrong results due to parenthesis, might need better handling
                        label = "|".join(
                            sorted([t.symbol for t in transitions if t.symbol != ""])
                        )

                    if "" in symbols and symbols != {""}:
                        if len(label) == 1:
                            label += "?"
                        else:
                            label = f"(?:{label})?"
                    self.set_transition(label, s, s2)

        return self

    def to_regex(self):
        """
        Convert the NFA to a regex using state elimination
        :return:
        """
        copy_nfa = copy.deepcopy(self.simplify())
        copy_nfa.simplify()._fix()

        copy_nfa._merge_parallel_transitions()

        # Eliminate states until only 2 states are left
        while len(copy_nfa.states) > 2:
            # find a state to eliminate
            # sort all states by the number of transitions
            sorted_states = sorted(
                copy_nfa.states,
                key=lambda x: len(x.get_transitions())
                + len(x.get_transitions(direction=Direction.BACKWARD)),
            )
            # the state should not be the start state or the end state
            state_to_eliminate = next(
                filter(
                    lambda x: not x.is_end_state and x != copy_nfa.start_state,
                    sorted_states,
                )
            )

            # get all In and Out states
            #  if In state == Out state ( == State to eliminate) => (transition)*
            out_transitions = state_to_eliminate.get_transitions()
            in_transitions = state_to_eliminate.get_transitions(
                direction=Direction.BACKWARD
            )
            out_states = [t.to_state for t in out_transitions]
            in_states = [t.from_state for t in in_transitions]

            self_loop_transition_label = ""
            # Check if the state to eliminate has a self-loop
            if state_to_eliminate in out_states and state_to_eliminate in in_states:
                self_loop_transition = next(
                    t for t in out_transitions if t.to_state == state_to_eliminate
                )
                # Get symbol of self-loop transition
                self_loop_transition_label = self_loop_transition.symbol
                # if the symbol has a length > 1, we need to add parentheses
                if len(self_loop_transition_label) > 1:
                    self_loop_transition_label = f"(?:{self_loop_transition_label})*"
                else:
                    self_loop_transition_label += "*"
                # Remove self-loop
                copy_nfa.remove_transition(self_loop_transition)

            # Eliminate state
            for t in list(out_transitions):
                for t2 in list(in_transitions):
                    if t2.to_state is not t.to_state:
                        # Create a new transition
                        # FIXME this is super hacky, it could break in the future
                        first_transition_label = (
                            f"(?:{t2.symbol})" if "|" in t2.symbol else t2.symbol
                        )
                        second_transition_label = (
                            f"(?:{t.symbol})" if "|" in t.symbol else t.symbol
                        )
                        copy_nfa.set_transition(
                            first_transition_label
                            + self_loop_transition_label
                            + second_transition_label,
                            t2.from_state,
                            t.to_state,
                        )

            # Remove old transitions
            for t in in_transitions + out_transitions:
                copy_nfa.remove_transition(t)

            copy_nfa.states.remove(state_to_eliminate)

            copy_nfa._merge_parallel_transitions()

        # Get the final transition
        final_transition = copy_nfa.transitions[0]
        return final_transition.symbol

    def _stitch_synched_nfa(self, synched_nfa):
        # Warning: this currently assumes that the synced NFA was generated by the self NFA and another NFA
        # With self being the first NFA in the sync

        # States to substitute
        substitution_states = {}

        # Stitch states
        for s in synched_nfa.states:
            substitution_states[s.states[0]] = s

            # Add the sync state to the self NFA, but as a State instead of a SynchronizedState
            self.states.append(s)

            # if the synched state is an exit state, connect the sync state to the states[0] state
            if s.is_exit_state():
                self.set_epsilon_transition(s, s.states[0])
            
            # if the synched state is an entry state, connect the states[0] state to the sync state
            if s.is_entry_state():
                self.set_epsilon_transition(s.states[0], s)

        # Add new synched transitions to self NFA
        self.transitions.extend(synched_nfa.transitions)

        return self

    def _merge_word_boundary(self, boundary):
        word_boundary_nfa = from_regex(r"\w").simplify()
        non_word_boundary_nfa = from_regex(r"\W").simplify()
        # nfa
        joint_state = boundary.from_state

        # \w\W
        backward_merged_nfa = self.merge(
            word_boundary_nfa,
            nfa1_sync_state=joint_state,
            direction=Direction.BACKWARD,
        )

        forward_merged_nfa = self.merge(
            non_word_boundary_nfa,
            nfa1_sync_state=joint_state,
            direction=Direction.FORWARD,
        )

        word_nonword_nfa = None

        if backward_merged_nfa and forward_merged_nfa:
            # Concatenate the \w and \W synched NFAs
            word_nonword_nfa = backward_merged_nfa.concatenate(
                forward_merged_nfa,
                # get synced states that contain the joint_state_minus_one as the first state
                on_end_states=[
                    s
                    for s in backward_merged_nfa.states
                    if s.is_exit_state()
                ],
                on_other_nfa_start_states=[
                    s
                    for s in forward_merged_nfa.states
                    if s.is_entry_state()
                ],
            )

        # Create new boundary NFAs to avoid duplicates in synced states
        # TODO check if this actually matters
        non_word_boundary_nfa = from_regex(r"\W").simplify()
        word_boundary_nfa = from_regex(r"\w").simplify()

        # \W\w
        backward_merged_nfa = self.merge(
            non_word_boundary_nfa,
            nfa1_sync_state=joint_state,
            direction=Direction.BACKWARD,
        )
        forward_merged_nfa = self.merge(
            word_boundary_nfa,
            nfa1_sync_state=joint_state,
            direction=Direction.FORWARD,
        )

        nonword_word_nfa = None

        if backward_merged_nfa and forward_merged_nfa:
            # Concatenate the \W and \w synched NFAs
            nonword_word_nfa = backward_merged_nfa.concatenate(
                forward_merged_nfa,
                # get synced states that contain the joint_state_minus_one as the first state
                on_end_states=[
                    s
                    for s in backward_merged_nfa.states
                    if s.is_exit_state()
                ],
                on_other_nfa_start_states=[
                    s
                    for s in forward_merged_nfa.states
                    if s.is_entry_state()
                ],
            )

        # Stitch concatenated NFA to the original NFA
        # FIXME this is currently stitched at the end because _stitch is destructive for the original NFA
        #   If we stitch before, we might break the merge process for the second boundary NFA
        if word_nonword_nfa:
            self._stitch_synched_nfa(word_nonword_nfa)
            # Transfer other boundaries to the original NFA
            # (excluding the one we're currently processing)
            for b in word_nonword_nfa.boundaries:
                if b.from_state.states[0] != joint_state:
                    self.boundaries.append(b)

        if nonword_word_nfa:
            self._stitch_synched_nfa(nonword_word_nfa)
            # Transfer boundaries to the original NFA
            # (excluding the one we're currently processing)
            for b in nonword_word_nfa.boundaries:
                if b.from_state.states[0] != joint_state:
                    self.boundaries.append(b)

        # Remove the original boundary
        self.boundaries.remove(boundary)
        self.remove_state(joint_state)

        return self
    
    def flatten_lookarounds(self):
        """
        Flatten lookarounds in the NFA
        # TODO implement it for other lookarounds, currently only word boundaries
        :return:
        """
        while self.boundaries:
            b = self.boundaries[0]
            if b.from_state not in self:
                self.boundaries.remove(b)
                continue
            self._merge_word_boundary(b).simplify()
        
        return self



def from_regex(regex):
    parsed = sre_parse.parse(regex)
    return from_regex_pattern(parsed)


def from_regex_pattern(regex: sre_parse.SubPattern):
    if 0 == len(regex.data):
        raise ValueError("ERROR: regex is empty")
    elif 1 == len(regex.data):
        return sre_pattern_to_nfa(regex[0])
    else:
        nfas = [sre_pattern_to_nfa(construct) for construct in regex.data]
        for n in nfas[1:]:
            nfas[0].concatenate(n)
        return nfas[0]


def sre_pattern_to_nfa(pattern):
    node_type, node_value = pattern
    if sre_constants.LITERAL == node_type:  # a
        return NFA().append_transition(chr(node_value))
    elif sre_constants.NOT_LITERAL == node_type:  # [^a]
        positive_range = NFA().append_transition(chr(node_value))
        return positive_range.negate_range_transition_between(
            positive_range.start_state, positive_range.get_end_states()[0]
        )
    elif (
        sre_constants.RANGE == node_type
    ):  # # [abc], but also (a|b|c) is translated to this
        low, high = node_value
        return alternate(
            *[NFA().append_transition(chr(i)) for i in range(low, high + 1)]
        )
    elif sre_constants.SUBPATTERN == node_type:  # (a)
        # FIXME: we need to address the usage of backreferences and captured groups, currently ignored
        return from_regex_pattern(node_value[-1])
    # FIXME: min_repeat = max_repeat is very very very wrong and could lead to unexpected results
    # Using it as a temporary workaround to start testing
    elif sre_constants.MAX_REPEAT == node_type or sre_constants.MIN_REPEAT == node_type:
        low, high, value = node_value
        if (0, 1) == (low, high):  # a?
            return from_regex_pattern(value).make_skippable()
        elif (0, sre_constants.MAXREPEAT) == (low, high):  # a*
            return from_regex_pattern(value).make_kleene()
        elif (1, sre_constants.MAXREPEAT) == (low, high):  # a+
            return from_regex_pattern(value).concatenate(
                from_regex_pattern(value).make_kleene()
            )
        else:  # a{3,5}, a{3}
            nfa = NFA()
            for _ in range(low):
                nfa.concatenate(from_regex_pattern(value))
            if high == sre_constants.MAXREPEAT:
                nfa.concatenate(from_regex_pattern(value).make_kleene())
            else:
                for _ in range(high - low):
                    nfa.concatenate(from_regex_pattern(value).make_skippable())
            return nfa
    elif sre_constants.BRANCH == node_type:  # ab|cd
        _, value = node_value
        return alternate(*[from_regex_pattern(v) for v in value])
    elif sre_constants.IN == node_type:  # [abc], but also (a|b|c) is translated to this
        first_subnode_type, _ = node_value[0]
        if sre_constants.NEGATE == first_subnode_type:  # [^abc]
            positive_range = alternate(
                *[sre_pattern_to_nfa(subpattern) for subpattern in node_value[1:]]
            ).simplify()
            return positive_range.negate_range_transition_between(
                positive_range.start_state, positive_range.get_end_states()[0]
            )
        else:
            return alternate(
                *[sre_pattern_to_nfa(subpattern) for subpattern in node_value]
            ).simplify()
    elif sre_constants.ANY == node_type:  # .
        return alternate(*[NFA().append_transition(c) for c in utils.ALPHABET])
    elif sre_constants.CATEGORY == node_type:  # \d, \s, \w
        if sre_constants.CATEGORY_DIGIT == node_value:  # \d
            return from_regex("[0-9]")
        elif sre_constants.CATEGORY_NOT_DIGIT == node_value:  # \D
            return from_regex("[^0-9]")
        elif sre_constants.CATEGORY_SPACE == node_value:  # \s
            return from_regex("[ \t\n\r\f\v]")
        elif sre_constants.CATEGORY_NOT_SPACE == node_value:  # \S
            return from_regex("[^ \t\n\r\f\v]")
        elif sre_constants.CATEGORY_WORD == node_value:  # \w
            return from_regex("[a-zA-Z0-9_]")
        elif sre_constants.CATEGORY_NOT_WORD == node_value:  # \W
            return from_regex("[^a-zA-Z0-9_]")
        else:
            raise NotImplementedError(
                f"ERROR: regex category {node_value} not implemented"
            )
    # Lookarounds
    elif node_type in [
        sre_constants.ASSERT_NOT,
        sre_constants.ASSERT,
    ]:  # (?!abc), (?<!abc), (?=abc), (?<=abc)
        direction, pattern = node_value
        base_nfa = NFA()
        lookaround_nfa = from_regex_pattern(pattern)
        lookaround = Lookaround(
            base_nfa.start_state,
            ENUM_LOOKAROUND_TYPE[node_type],
            # TODO refactor using Direction enum
            ENUM_LOOKAROUND_DIRECTION[direction],
            lookaround_nfa,
        )
        base_nfa.lookarounds.append(lookaround)
        return base_nfa
    # Boundaries
    elif node_type == sre_constants.AT:  # \b, \B
        boundary = node_value
        base_nfa = NFA()
        boundary = Boundary(base_nfa.start_state, boundary)
        base_nfa.boundaries.append(boundary)
        return base_nfa

    else:
        raise NotImplementedError(f"ERROR: regex construct {pattern} not implemented")


def alternate(*nfas):
    """
    Create a new NFA that accepts the union of the strings accepted by the input NFAs
    Basically an OR for NFAs
    :param nfas:
    :return:
    """
    nfa = NFA()

    # Mark initial state as non-accepting and create a new end state
    nfa.start_state.is_end_state = False
    new_end_state = State()
    nfa.states.append(new_end_state)

    for n in nfas:
        nfa.states.extend(n.states)
        nfa.transitions.extend(n.transitions)
        nfa.set_epsilon_transition(nfa.start_state, n.start_state)
        for end_state in n.get_end_states():
            end_state.is_end_state = False
            nfa.set_epsilon_transition(end_state, new_end_state)

    return nfa


def negate(nfa_orig):
    """
    Negate an NFA by creating a new NFA that accepts all strings not accepted by the input NFA
    :param nfa_orig:
    :return:
    """
    nfa_orig_copy = copy.deepcopy(nfa_orig.simplify())
    nfa = copy.deepcopy(nfa_orig_copy)

    nfa.start_state.is_end_state = False
    continue_end_state = State()
    # empty_end_state = State()
    for s in nfa.states:
        s.is_end_state = not s.is_end_state
        if s.is_end_state:
            # nfa.set_epsilon_transition(s, empty_end_state)
            # print(s.id)
            # print([(t.to_state.id, t.symbol) for t in s.get_transitions()])
            # print(sorted(s.get_all_outgoing_symbols()))
            # print(sorted(s.complement_outbound_transitions()))
            for symbol in s.complement_outbound_transitions():
                nfa.set_transition(symbol, s, continue_end_state)
            # s.is_end_state = False

    nfa.states.append(continue_end_state)
    nfa.concatenate(from_regex(".*"), on_states=[continue_end_state])

    # nfa.states.append(empty_end_state)
    return nfa


def intersect(nfa1: NFA, nfa2: NFA, nfa1_sync_state=None, nfa2_sync_state=None):
    """
    Intersect two NFAs by creating a new NFA that accepts strings accepted by both input NFAs
    Basically an AND for NFAs

    :param nfa1:
    :param nfa2:
    :param nfa1_sync_state: optional state in nfa1 to sync on
    :param nfa2_sync_state: optional state in nfa2 to sync on
    :return:
    """

    nfa1.simplify()
    nfa2.simplify()

    merged_nfa = nfa1.merge(nfa2)
    stitched_nfa = nfa1._stitch_synched_nfa(merged_nfa)

    return stitched_nfa



def main1():
    nfa = from_regex(r".\b.\b.").simplify()
    nfa = from_regex(r".*\balert\b.*").simplify()
    nfa.to_dot(view=True, simplified=True)
    input()

    nfa.flatten_lookarounds()

    nfa.to_dot(view=True)
    print(nfa.walk())

    print("debug")

if __name__ == "__main__":
    main1()
