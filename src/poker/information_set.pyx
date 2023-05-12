#!python
#cython: language_level=3

cdef class InformationSet:
    def __init__(self, str public_game_state, str player_private_cards, str betting_history):
        self.public_game_state = public_game_state
        self.player_private_cards = player_private_cards
        self.betting_history = betting_history

    def __eq__(self, other):
        if not isinstance(other, InformationSet):
            return False
        return (self.public_game_state == other.public_game_state
                and self.player_private_cards == other.player_private_cards
                and self.betting_history == other.betting_history)

    def clone(self):
        return InformationSet(self.public_game_state, self.player_private_cards, self.betting_history)

    def __hash__(self):
        return hash((self.public_game_state, self.player_private_cards, self.betting_history))
