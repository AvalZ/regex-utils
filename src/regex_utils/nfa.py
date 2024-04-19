import copy
import random
import sre_constants
import sre_parse

import graphviz
from graphviz.quoting import *

from regex_utils import nfautils


class State:
    id_counter = 0

    def __init__(self, end_state=True):
        self.id = State.id_counter
        State.id_counter += 1
        self.out_transitions = []
        self.in_transitions = []
        self.is_end_state = end_state

    def is_dead_end(self):
        return not self.is_end_state and (
            len(self.out_transitions) == 0
            or all(t.is_self_loop() for t in self.out_transitions)
        )

    def get_all_outgoing_symbols(self):
        symbols = set(t.symbol for t in self.out_transitions)

        states = {self}

        while True:
            previous_states = states.copy()

            for t in self.out_transitions:
                if t.is_epsilon_transition() and not t.is_self_loop():
                    symbols.update(t.to_state.get_all_outgoing_symbols())
                    states.add(t.to_state)

            if states == previous_states:
                break

        return symbols

    def complement_outbound_transitions(self):
        return set(NFA.alphabet) - self.get_all_outgoing_symbols()


class Transition:
    def __init__(self, symbol, from_state, to_state):
        self.symbol = symbol
        self.from_state = from_state
        self.to_state = to_state

    def is_self_loop(self):
        return self.from_state == self.to_state

    def is_epsilon_transition(self):
        return self.symbol == ""


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


class NFA:
    @staticmethod
    def from_regex(regex):
        parsed = sre_parse.parse(regex)
        return NFA.from_regex_pattern(parsed)

    @staticmethod
    def from_regex_pattern(regex: sre_parse.SubPattern):
        if 0 == len(regex.data):
            raise ValueError("ERROR: regex is empty")
        elif 1 == len(regex.data):
            return NFA.sre_pattern_to_nfa(regex[0])
        else:
            nfas = [NFA.sre_pattern_to_nfa(construct) for construct in regex.data]
            for n in nfas[1:]:
                nfas[0].concatenate(n)
            return nfas[0]

    @staticmethod
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
            return NFA.alternate(
                *[NFA().append_transition(chr(i)) for i in range(low, high + 1)]
            )
        elif sre_constants.SUBPATTERN == node_type:  # (a)
            # FIXME: we need to address the usage of backreferences and captured groups, currently ignored
            return NFA.from_regex_pattern(node_value[-1])
        # FIXME: min_repeat = max_repeat is very very very wrong and could lead to unexpected results
        # Using it as a temporary workaround to start testing
        elif (
            sre_constants.MAX_REPEAT == node_type
            or sre_constants.MIN_REPEAT == node_type
        ):
            low, high, value = node_value
            if (0, 1) == (low, high):  # a?
                return NFA.from_regex_pattern(value).make_skippable()
            elif (0, sre_constants.MAXREPEAT) == (low, high):  # a*
                return NFA.from_regex_pattern(value).make_kleene()
            elif (1, sre_constants.MAXREPEAT) == (low, high):  # a+
                return NFA.from_regex_pattern(value).concatenate(
                    NFA.from_regex_pattern(value).make_kleene()
                )
            else:  # a{3,5}, a{3}
                nfa = NFA()
                for _ in range(low):
                    nfa.concatenate(NFA.from_regex_pattern(value))
                if high == sre_constants.MAXREPEAT:
                    nfa.concatenate(NFA.from_regex_pattern(value).make_kleene())
                else:
                    for _ in range(high - low):
                        nfa.concatenate(NFA.from_regex_pattern(value).make_skippable())
                return nfa
        elif sre_constants.BRANCH == node_type:  # ab|cd
            _, value = node_value
            return NFA.alternate(*[NFA.from_regex_pattern(v) for v in value])
        elif (
            sre_constants.IN == node_type
        ):  # [abc], but also (a|b|c) is translated to this
            first_subnode_type, _ = node_value[0]
            if sre_constants.NEGATE == first_subnode_type:  # [^abc]
                positive_range = NFA.alternate(
                    *[
                        NFA.sre_pattern_to_nfa(subpattern)
                        for subpattern in node_value[1:]
                    ]
                ).simplify()
                return positive_range.negate_range_transition_between(
                    positive_range.start_state, positive_range.get_end_states()[0]
                )
            else:
                return NFA.alternate(
                    *[NFA.sre_pattern_to_nfa(subpattern) for subpattern in node_value]
                ).simplify()
        elif sre_constants.ANY == node_type:  # .
            return NFA.alternate(*[NFA().append_transition(c) for c in NFA.alphabet])
        elif sre_constants.CATEGORY == node_type:  # \d, \s, \w
            if sre_constants.CATEGORY_DIGIT == node_value:  # \d
                return NFA.from_regex("[0-9]")
            elif sre_constants.CATEGORY_NOT_DIGIT == node_value:  # \D
                return NFA.from_regex("[^0-9]")
            elif sre_constants.CATEGORY_SPACE == node_value:  # \s
                return NFA.from_regex("[ \t\n\r\f\v]")
            elif sre_constants.CATEGORY_NOT_SPACE == node_value:  # \S
                return NFA.from_regex("[^ \t\n\r\f\v]")
            elif sre_constants.CATEGORY_WORD == node_value:  # \w
                return NFA.from_regex("[a-zA-Z0-9_]")
            elif sre_constants.CATEGORY_NOT_WORD == node_value:  # \W
                return NFA.from_regex("[^a-zA-Z0-9_]")
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
            lookaround_nfa = NFA.from_regex_pattern(pattern)
            lookaround = Lookaround(
                base_nfa.start_state,
                ENUM_LOOKAROUND_TYPE[node_type],
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
            raise NotImplementedError(
                f"ERROR: regex construct {pattern} not implemented"
            )

    @staticmethod
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

    @staticmethod
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
                # print([(t.to_state.id, t.symbol) for t in s.out_transitions])
                # print(sorted(s.get_all_outgoing_symbols()))
                # print(sorted(s.complement_outbound_transitions()))
                for symbol in s.complement_outbound_transitions():
                    nfa.set_transition(symbol, s, continue_end_state)
                # s.is_end_state = False

        nfa.states.append(continue_end_state)
        nfa.concatenate(NFA.from_regex(".*"), on_states=[continue_end_state])

        # nfa.states.append(empty_end_state)
        return nfa

    @staticmethod
    def intersect(nfa1, nfa2):
        """
        Intersect two NFAs by creating a new NFA that accepts strings accepted by both input NFAs
        Basically an AND for NFAs

        :param nfa1:
        :param nfa2:
        :return:
        """

        nfa1.simplify()
        nfa2.simplify()
        nfa = NFA()
        nfa.start_state.is_end_state = (
            nfa1.start_state.is_end_state and nfa2.start_state.is_end_state
        )

        synchronized_states = {(nfa1.start_state, nfa2.start_state): nfa.start_state}

        while True:
            previous_synchronized_states = synchronized_states.copy()

            # print("Synched states", [f"{s.id}:({s1.id},{s2.id})" for (s1, s2), s in synchronized_states.items()])
            for (s1, s2), sync_state in dict(synchronized_states).items():
                # print(f"Processing synchronized states {s1.id} and {s2.id}")
                # Consume epsilon transitions in nfa2, otherwise it will miss some possible steps
                for t in s2.out_transitions:
                    if t.is_epsilon_transition():
                        if (s1, t.to_state) not in synchronized_states:
                            synchronized_states[(s1, t.to_state)] = State(
                                end_state=(s1.is_end_state and t.to_state.is_end_state)
                            )
                            nfa.states.append(synchronized_states[(s1, t.to_state)])
                        nfa.set_epsilon_transition(
                            sync_state, synchronized_states[(s1, t.to_state)]
                        )

                for t in s1.out_transitions:
                    # If there's an epsilon transition, consume it "for free"
                    # And add a new synchronized state
                    if t.is_epsilon_transition():
                        if (t.to_state, s2) not in synchronized_states:
                            synchronized_states[(t.to_state, s2)] = State(
                                end_state=(t.to_state.is_end_state and s2.is_end_state)
                            )
                            nfa.states.append(synchronized_states[(t.to_state, s2)])
                        nfa.set_epsilon_transition(
                            sync_state, synchronized_states[(t.to_state, s2)]
                        )

                    # Otherwise, check if the other nfa has transitions with the same symbol
                    # If it does, create a new synchronized state for each to_state of each transition
                    # And add a transition to the new synchronized state
                    for t2 in filter(
                        lambda t2: t2.symbol == t.symbol, s2.out_transitions
                    ):
                        if (t.to_state, t2.to_state) not in synchronized_states:
                            synchronized_states[(t.to_state, t2.to_state)] = State(
                                end_state=(
                                    t.to_state.is_end_state and t2.to_state.is_end_state
                                )
                            )
                            nfa.states.append(
                                synchronized_states[(t.to_state, t2.to_state)]
                            )
                        # print("Adding transition", t.symbol, "from", sync_state.id, "to", synchronized_states[(t.to_state, t2.to_state)].id)
                        nfa.set_transition(
                            t.symbol,
                            sync_state,
                            synchronized_states[(t.to_state, t2.to_state)],
                        )

            if len(synchronized_states) == len(previous_synchronized_states):
                break

        return nfa

    def concatenate(self, nfa, on_states=None):
        if on_states is None:
            end_states = self.get_end_states()
        else:
            end_states = on_states

        for s in end_states:
            self.set_epsilon_transition(s, nfa.start_state)
            s.is_end_state = False

        self.states.extend(nfa.states)
        self.transitions.extend(nfa.transitions)
        self.lookarounds.extend(nfa.lookarounds)
        self.boundaries.extend(nfa.boundaries)
        return self

    alphabet = nfautils.ALPHABET

    def __init__(self):
        # Create an empty NFA
        self.boundaries = []
        self.states = [State()]
        self.transitions = []
        self.start_state = self.states[0]
        self.current_state = self.start_state
        # Track lookarounds
        self.lookarounds = []

    def get_end_states(self):
        return list(filter(lambda x: x.is_end_state, self.states))

    def reset(self):
        self.current_state = self.start_state

    def step(self):
        if self.current_state.out_transitions:
            t = random.choice(self.current_state.out_transitions)
            self.consume(t)
            return t.symbol
        else:
            return None

    def walk(self, continue_after_end_state=0.0):
        # TODO: add new walk techniques, e.g., weight walk on transitions that were already taken (to avoid being stuck in loops)
        self.reset()

        symbols = []
        while True:  # Otherwise it will stop at empty strings (epsilon-transitions)
            symbols.append(self.step())
            if self.current_state.is_end_state:
                # Avoid generating a random number if continue_after_end_state is set to 0
                if (
                    continue_after_end_state == 0.0
                    or random.random() > continue_after_end_state
                    or not self.current_state.out_transitions
                ):
                    break

        return "".join(symbols)

    def consume(self, transition):
        self.current_state = transition.to_state

    def append_transition(self, symbol):
        s = State()
        t = Transition(symbol, self.current_state, s)
        t.from_state.out_transitions.append(t)
        t.to_state.in_transitions.append(t)
        # Add transition and new state to the NFA
        self.transitions.append(t)
        self.states.append(s)
        self.current_state.is_end_state = False
        self.consume(t)
        return self

    def append_epsilon_transition(self):
        return self

    def set_transition(self, symbol, from_state, to_state):
        # check if there's another transition between these two states with the same symbol
        existing_transition = next(
            (
                t
                for t in from_state.out_transitions
                if t.symbol == symbol and t.to_state == to_state
            ),
            None,
        )
        if existing_transition:
            return existing_transition

        # Otherwise, create a new transition
        t = Transition(symbol, from_state, to_state)
        self.transitions.append(t)
        from_state.out_transitions.append(t)
        to_state.in_transitions.append(t)
        return t

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
        for t in removed_state.out_transitions:
            t.from_state = merged_state
            merged_state.out_transitions.append(t)

        for t in removed_state.in_transitions:
            t.to_state = merged_state
            merged_state.in_transitions.append(t)

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

        for l in lookaheads_on_removed_state:
            l.from_state = merged_state

        for b in boundaries_on_removed_state:
            b.from_state = merged_state

        try:
            self.states.remove(removed_state)
        except ValueError:
            print("Cannot remove state", removed_state.id)

        return self

    def naive_merge_states(self, merged_state, removed_state):
        if merged_state is removed_state:
            return self

        # print("Merging states", merged_state.id, removed_state.id)
        # Transfer all transitions from to_state to from_state
        for t in removed_state.out_transitions:
            # Check if there's another identical transition
            # With the same symbol, to_state and from_state
            # print("for t in removed_state.out_transitions:")
            if any(
                filter(
                    lambda x: x.symbol == t.symbol
                    and x.to_state == t.to_state
                    and x.from_state == merged_state,
                    merged_state.out_transitions,
                )
            ):
                self.remove_transition(t)
                continue
            t.from_state = merged_state
            if t not in merged_state.out_transitions:
                merged_state.out_transitions.append(t)

        for t in removed_state.in_transitions:
            # Check if there's another identical transition
            # With the same symbol, to_state and from_state
            if any(
                filter(
                    lambda x: x.symbol == t.symbol
                    and x.to_state == t.to_state
                    and x.from_state == merged_state,
                    merged_state.in_transitions,
                )
            ):
                try:
                    self.remove_transition(t)
                except ValueError:
                    pass
            t.to_state = merged_state
            if t not in merged_state.in_transitions:
                merged_state.in_transitions.append(t)

        # Remove epsilon transitions from from_state to to_state
        # Notice that, due to the above for loop
        # transitions between from_state and to_state are now self-loops
        for t in merged_state.out_transitions:
            if t.from_state == t.to_state and t.symbol == "":
                self.remove_transition(t)

        if removed_state.is_end_state:
            merged_state.is_end_state = True

        if removed_state is self.start_state:
            self.start_state = merged_state

        lookaheads_on_removed_state = list(
            filter(lambda x: x.from_state == removed_state, self.lookarounds)
        )
        for l in lookaheads_on_removed_state:
            l.from_state = merged_state

        # Remove to_state from the list of states
        try:
            self.states.remove(removed_state)
        except ValueError:
            print("Cannot remove state", removed_state.id)
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
            previous_states = len(self.states)
            previous_transitions = len(self.transitions)
            previous_end_states = len(
                list(filter(lambda x: x.is_end_state, self.states))
            )

            self.remove_useless_epsilon_transitions()
            self.remove_two_way_epsilon_transitions()
            self.remove_epsilon_self_loops()

            self.remove_dead_end_states()
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
                    print("Simplify reached budget")
                break

        self.rewrite_states_ids()
        return self

    def propagate_end_states(self):
        # Propagate end states
        # If there is an epsilon transition from a state to an end state,
        # mark the state as an end state
        for s in self.states:
            if any(
                filter(
                    lambda x: x.is_epsilon_transition() and x.to_state.is_end_state,
                    s.out_transitions,
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
            for t in s.out_transitions:
                if t.is_epsilon_transition() and not t.is_self_loop():
                    for t2 in t.to_state.out_transitions:
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
                    len(t.from_state.out_transitions) == 1
                    or len(t.to_state.in_transitions) == 1
                ):
                    # Or if there is another opposite epsilon transition
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
                for t in s.in_transitions:
                    self.remove_transition(t)

            # Exit if it didn't remove any states, otherwise repeat
            # since removing transitions could have created new dead-end states
            if previous_states == len(self.states):
                return self

    def remove_duplicate_transitions(self):
        # Remove duplicate transitions
        for s in self.states:
            # Group transitions by symbol and to_state
            for t in s.out_transitions:
                duplicate_transitions = [
                    t2
                    for t2 in s.out_transitions
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
        not_symbols = list(set(NFA.alphabet) - set(symbols))

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
            transition.from_state.out_transitions.remove(transition)
        except ValueError:
            pass

        try:
            transition.to_state.in_transitions.remove(transition)
        except ValueError:
            pass

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
                color="red"
                if any(filter(lambda x: x.from_state == s, self.lookarounds))
                else "black",
            )

        # Group transitions by from_state and to_state
        # If from_state and to_state are the same for multiple transitions,
        # group in a single edge and show a range instead as a label
        excluded_transitions = []

        for t in self.transitions:
            if t in excluded_transitions:
                continue
            # print("- Drawing round for transition ", t.from_state.id, t.to_state.id, t.symbol)
            symbols = set()
            same_transitions = self._get_parallel_transitions(
                t, get_epsilon_transitions=True
            )
            if same_transitions:
                symbols.update(t.symbol for t in same_transitions)

            if "" in symbols:
                # print("Creating Transition", t.from_state.id, t.to_state.id, "ε")
                dot.edge(str(t.from_state.id), str(t.to_state.id), label="ε")
                symbols = symbols - {""}

            if t.symbol:
                # print("Add symbol", t.symbol)
                symbols.add(t.symbol)

            label = nfautils.range_label(symbols)

            if label:
                # ("Creating transition", t.from_state.id, t.to_state.id, repr(label))
                dot.edge(
                    str(t.from_state.id),
                    str(t.to_state.id),
                    label=escape(repr(label))[1:-1],
                )

            for st in same_transitions:
                # print("Removing transition", st.from_state.id, st.to_state.id, st.symbol)
                excluded_transitions.append(st)

        # Workaround to mark start state
        dot.node("", shape="none", width="0")
        dot.edge("", str(self.start_state.id))

        dot.render(view=view)

        return dot.source

    def rewrite_states_ids(self):
        for i, s in enumerate(self.states):
            s.id = i

        return self

    def describe(self):
        for s in self.states:
            # Print state id and check if it's the start state
            print(f"State {s.id} is_end_state={s.is_end_state}")
            if s == self.start_state:
                print("\tStart state")
            for t in sorted(s.in_transitions, key=lambda x: x.symbol):
                print(f"\t {t.from_state.id} -{t.symbol}->")
            for t in sorted(s.out_transitions, key=lambda x: x.symbol):
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
            for s2 in [t.to_state for t in s.out_transitions]:
                transitions = self._get_transitions_between_states(
                    s, s2, get_epsilon_transitions=True
                )

                if transitions:
                    for t in transitions:
                        self.remove_transition(t)
                    # print("Found parallel transitions between", s.id, "and", s2.id)
                    # Create a new transition
                    symbols = set(t.symbol for t in transitions)
                    # print("Symbols", symbols)
                    if all(len(symbol) == 1 or not symbol for symbol in symbols):
                        label = nfautils.range_label(symbols - {""})
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
                    # print("Merging transitions", "|".join([t.symbol for t in transitions]), "from", s.id, "to", s2.id)
                    # Remove old transitions
                    # print("Out transitions", s.out_transitions)

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
                key=lambda x: len(x.out_transitions) + len(x.in_transitions),
            )
            # the state should not be the start state or the end state
            state_to_eliminate = next(
                filter(
                    lambda x: not x.is_end_state and x != copy_nfa.start_state,
                    sorted_states,
                )
            )

            # print("Eliminating state", state_to_eliminate.id, "with", len(state_to_eliminate.out_transitions), "out transitions", "and", len(state_to_eliminate.in_transitions), "in transitions")
            # print("\tOut transitions", [(t.symbol, t.to_state.id) for t in state_to_eliminate.out_transitions])
            # print("\tIn transitions", [(t.symbol, t.from_state.id) for t in state_to_eliminate.in_transitions])
            # get all In and Out states
            #  if In state == Out state ( == State to eliminate) => (transition)*
            out_transitions = state_to_eliminate.out_transitions
            in_transitions = state_to_eliminate.in_transitions
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
                        # print("Creating transition", t2.symbol + self_loop_transition_label + t.symbol, "from", t2.from_state.id, "to", t.to_state.id)
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


if __name__ == "__main__":
    regex1 = "ab[a-z]"
    regex2 = "abc*"

    nfa = NFA.from_regex(regex1)
    # nfa = NFA.negate(nfa)

    nfa2 = NFA.from_regex(regex2)

    nfa = NFA.intersect(nfa, nfa2).simplify()
    resulting_regex = nfa.to_regex()
    print(resulting_regex)
    nfa.to_dot(simplified=True)

    values = set()
    for _ in range(10):
        random_walk = nfa.walk(continue_after_end_state=0.2)
        values.add(repr(random_walk))

    # resulting_regex = nfa.to_regex()

    for random_walk in sorted(values):
        # assert re.match(resulting_regex, random_walk)
        print(random_walk)

    # print("Resulting regex:", resulting_regex)
