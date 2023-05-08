cdef class GameState:
   
    cdef public list players
    cdef public int small_blind
    cdef public int big_blind
    cdef public int dealer_position
    cdef public int pot
    cdef public int current_bet
    cdef public unsigned long long board
    cdef public list deck

    cdef void fisher_yates_shuffle(self)

    cpdef draw_card(self)

    cpdef deal_cards(self)

    cpdef reset(self)

cpdef unsigned long long card_to_int(str suit, str value)

cpdef public list create_deck()

cpdef str int_to_card(unsigned long long card)

cpdef unsigned long long card_str_to_int(str card_str)

cpdef str format_hand(unsigned long long hand)

cpdef display_game_state(GameState game_state, int player_index)