from ._utils cimport *
from .player cimport Player

cdef class GameState:
    cdef public int hand_id
    cdef public int silent

    cdef public list suits
    cdef public list values
   
    cdef public list positions
    cdef public list[Player] players

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
    cdef public Deck deck

    cdef public list betting_history

    cdef void reset(self)
    cpdef GameState clone(self)
    cdef void assign_positions(self)
    cdef void update_current_hand(self, object hand)
    cdef void handle_blinds(self)
    cpdef void setup_preflop(self, object hand = *)
    cpdef void setup_postflop(self, str round_name)
    cpdef bint handle_action(self, object action = *)
    cdef str abstract_hand(self, unsigned long long card1, unsigned long long card2)
    cdef void progress_to_showdown(self)
    cdef void showdown(self)
    cdef bint is_terminal(self)
    cdef bint is_terminal_river(self)
    cpdef void deal_private_cards(self, int player_index = *)
    cdef list generate_positions(self, int num_players)
    cdef int active_players(self)
    cdef int folded_players(self)
    cdef int allin_players(self)
    cdef void draw_card(self)
    cdef bint board_has_five_cards(self)
    cdef int num_board_cards(self)
    cpdef void debug_output(self)
    cdef void log_current_hand(self, object terminal = *)
    cpdef remove_str_card_from_deck(self, str card)

