#!python
#cython: language_level=3

from libc.stdlib cimport rand, srand
from .player cimport Player
from .ai_player cimport AIPlayer
from .poker_game cimport PokerGame, GameState, card_to_int, create_deck, fisher_yates_shuffle, draw_card, deal_cards, int_to_card, card_str_to_int, player_action, showdown, get_user_input, preflop, flop, turn, river, format_hand, display_game_state, process_user_input

cimport numpy
cimport cython

cdef class CFRTrainer:
    cdef public int iterations

    def __init__(self, int iterations):
        self.iterations = iterations

    cpdef train(self):
        # Main loop for the training iterations
        for _ in range(self.iterations):
            pass  # Implement the CFR algorithm here

    cdef double traverse_game_tree(self, GameState game_state, int player_index, double probability):
        pass  # Implement the game tree traversal and return the counterfactual value

    cdef double cfr(self, GameState game_state, int player_index, double probability):
        pass  # Implement the Counterfactual Regret Minimization algorithm

