cdef class GameState:
   
    cdef public list players
    cdef public int small_blind
    cdef public int big_blind
    cdef public int dealer_position
    cdef public int pot
    cdef public int current_bet
    cdef public int board

    cpdef reset(self)

cpdef unsigned long long card_to_int(str suit, str value)

cdef void fisher_yates_shuffle(list deck)

cpdef unsigned long long draw_card(list deck)

cpdef public list create_deck()

cpdef deal_cards(list deck, GameState game_state)

cpdef str int_to_card(unsigned long long card)

cpdef unsigned long long card_str_to_int(str card_str)

cpdef str format_hand(unsigned long long hand)

cpdef display_game_state(GameState game_state, int player_index)