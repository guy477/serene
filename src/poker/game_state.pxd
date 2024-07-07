from ._utils cimport *
from .player cimport Player


cdef class GameState:
    cdef int hand_id
    cdef int silent

    cdef list suits
    cdef list values
   
    cdef list positions
    cdef list[Player] players

    cdef int cur_round_index

    cdef int small_blind
    cdef int big_blind

    cdef int num_simulations
    
    cdef int player_index

    cdef int round_active_players
    cdef int num_actions
    cdef int dealer_position
    cdef int last_raiser
    cdef int current_bet

    cdef int winner_index
    cdef int pot
    
    cdef unsigned long long board
    cdef public Deck deck

    cdef list betting_history



###### Game logic

    cpdef void setup_preflop(self, object hand = *)

    cdef void assign_positions(self)

    cdef void handle_blinds(self)
    
    cdef void deal_private_cards(self)

    cpdef void setup_postflop(self, str round_name)

    cpdef bint handle_action(self, object action = *, object strategy_trainer = *)

    cpdef bint step(self, action)

    cdef void showdown(self)
    

###### Utility

    cdef void progress_to_showdown(self)
    cdef bint is_terminal(self)
    cdef bint is_terminal_river(self)
    cdef Player get_current_player(self)
    cdef list generate_positions(self, int num_players)
    cdef int active_players(self)
    cdef int folded_players(self)
    cdef int allin_players(self)
    cdef void draw_card(self)
    cdef bint board_has_five_cards(self)
    cdef int num_board_cards(self)
    cdef void assign_positions(self)

    ### CFR
    cdef void update_current_hand(self, object hand)  
    cdef str abstract_hand(self, unsigned long long card1, unsigned long long card2)
    
    ### self
    cdef void reset(self)
    cpdef GameState clone(self)

###### Debug
    cpdef void debug_output(self)
    cdef void log_current_hand(self, object terminal = *)



