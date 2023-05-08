from .player cimport Player
from .ai_player cimport AIPlayer
from .game_state cimport GameState, card_to_int, create_deck, int_to_card, card_str_to_int, format_hand, display_game_state


cdef class PokerGame:
    cpdef public GameState game_state
    cpdef public list players
    cpdef public list deck
    cpdef play_game(self, int num_hands=*)
