cdef class GameState:
    cdef public int hand_id
    cdef public int silent

    cdef public list suits
    cdef public list values
   
    cdef public list positions
    cdef public list players

    cdef public int cur_round_index

    cdef public int small_blind
    cdef public int big_blind

    cdef public int num_simulations
    
    cdef public int player_index

    cdef public int round_active_players
    cdef public int num_actions
    cdef public int dealer_position
    cdef public int last_raiser
    cdef public int current_bet

    cdef public int winner_index
    cdef public int pot
    
    cdef public unsigned long long board
    cdef public list deck

    cdef public list betting_history

    cpdef reset(self)
    cpdef load_custom_betting_history(self, int round, object history)
    cpdef clone(self)
    cpdef assign_positions(self)
    cpdef handle_blinds(self)
    cpdef setup_preflop(self, object hand = *)
    cpdef str abstract_hand(self, unsigned long long card1, unsigned long long card2)
    cpdef setup_postflop(self, str round_name)
    cpdef bint handle_action(self, object action = *)
    cpdef progress_to_showdown(self)
    cpdef showdown(self)
    cpdef bint is_terminal(self)
    cpdef bint is_terminal_river(self)
    cpdef deal_private_cards(self, object hand = *)
    cpdef list generate_positions(self, int num_players)
    cdef void fisher_yates_shuffle(self)
    cpdef int active_players(self)
    cpdef int folded_players(self)
    cpdef int allin_players(self)
    cpdef draw_card(self)
    cpdef bint board_has_five_cards(self)
    cpdef int num_board_cards(self)
    cpdef debug_output(self)
    cpdef list create_deck(self, list suits, list values)
    cpdef log_current_hand(self, object terminal = *)

# 'Global' functions
cpdef unsigned long long card_to_int(str suit, str value)
cpdef public list create_deck()
cpdef str int_to_card(unsigned long long card)
cpdef unsigned long long card_str_to_int(str card_str)
cpdef list hand_to_cards(unsigned long long hand)
cpdef str format_hand(unsigned long long hand)
cpdef list hand_to_cards(unsigned long long hand)
cpdef display_game_state(GameState game_state, int player_index)
cdef unsigned int cy_evaluate(unsigned long long cards, unsigned int num_cards) nogil
cpdef cy_evaluate_cpp(cards, num_cards)

