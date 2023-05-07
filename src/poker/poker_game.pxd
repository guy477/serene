from .player cimport Player
from .ai_player cimport AIPlayer

cdef unsigned int cy_evaluate(unsigned long long cards, unsigned int num_cards) nogil
cdef class PokerGame:
    cpdef public GameState game_state
    cpdef public list players
    cpdef public list deck
    cpdef play_game(self, int num_hands=*)


cdef class GameState:
    cdef public int current_bet
    cdef public unsigned long long board
    cdef public list players
    cdef public int pot
    cdef public int dealer_position
    cdef public int small_blind
    cdef public int big_blind

    cpdef reset(self)

cpdef unsigned long long card_to_int(str suit, str value)

cpdef list create_deck()

cdef void fisher_yates_shuffle(list deck)

cpdef unsigned long long draw_card(list deck)

cpdef deal_cards(list deck, GameState game_state)

cpdef str int_to_card(unsigned long long card)

cpdef unsigned long long card_str_to_int(str card_str)

cpdef showdown(GameState game_state)

cdef handle_blinds(GameState game_state)

cpdef preflop(GameState game_state, list deck)

cpdef flop(GameState game_state, list deck)

cpdef turn(GameState game_state, list deck)

cpdef river(GameState game_state, list deck)

cpdef str format_hand(unsigned long long hand)

cpdef display_game_state(GameState game_state, int player_index)

cpdef player_action(GameState game_state, int player_index, str action, int bet_amount=*)

cpdef process_user_input(GameState game_state, int player_index)

cpdef str get_user_input(prompt)