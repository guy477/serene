from .player cimport Player
from .ai_player cimport AIPlayer
from .game_state cimport GameState, card_to_int, fisher_yates_shuffle, draw_card, create_deck, deal_cards, int_to_card, card_str_to_int, format_hand, display_game_state

cdef unsigned int cy_evaluate(unsigned long long cards, unsigned int num_cards) nogil
cdef class PokerGame:
    cpdef public GameState game_state
    cpdef public list players
    cpdef public list deck
    cpdef play_game(self, int num_hands=*)


cpdef showdown(GameState game_state)

cdef handle_blinds(GameState game_state)

cpdef preflop(GameState game_state, list deck)

cpdef postflop(GameState game_state, list deck, str round_name)
