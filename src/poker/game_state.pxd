cdef class GameState:
   
    cdef public list players
    cdef public int small_blind
    cdef public int big_blind
    cdef public int dealer_position
    cdef public int player_index
    cdef public int pot
    cdef public int current_bet
    cdef public unsigned long long board
    cdef public list deck


    # cdef get_current_player(self)

    # cdef get_information_set(self, int player)

    # cdef get_available_actions(self)

    cpdef reset(self)

    cpdef clone(self)

    cpdef handle_blinds(self)

    cpdef preflop(self)
    
    cpdef postflop(self, str round_name)
    
    cpdef showdown(self)

    cpdef bint is_terminal(self)

    cpdef deal_private_cards(self)

    cpdef deal_public_cards(self)

    cdef void fisher_yates_shuffle(self)

    cpdef int active_players(self)
    
    cpdef draw_card(self)

    cdef bint board_has_five_cards(self)


    


cpdef unsigned long long card_to_int(str suit, str value)

cpdef public list create_deck()

cpdef str int_to_card(unsigned long long card)

cpdef unsigned long long card_str_to_int(str card_str)

cpdef str format_hand(unsigned long long hand)

cpdef display_game_state(GameState game_state, int player_index)

cdef unsigned int cy_evaluate(unsigned long long cards, unsigned int num_cards) nogil

cpdef cy_evaluate_cpp(cards, num_cards)