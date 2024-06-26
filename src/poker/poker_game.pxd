from .player cimport Player
from .ai_player cimport AIPlayer
from .game_state cimport GameState, card_to_int, create_deck, int_to_card, card_str_to_int, format_hand, display_game_state
from .cfr cimport CFRTrainer

cdef class PokerGame:
    cdef public list suits
    cdef public list values
    cdef public logger
    cdef public GameState game_state
    cdef public list profit_loss
    cdef public dict position_pl
    cdef public list players
    cdef public list deck
    cpdef play_game(self, int num_hands=*)
    cpdef _play_round(self)
