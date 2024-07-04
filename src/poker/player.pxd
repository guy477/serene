from .game_state cimport GameState
from ._utils cimport *

cdef class Player:
    cdef public int chips
    cdef public list bet_sizing
    cdef public unsigned long long hand
    cdef public str abstracted_hand
    cdef public str position
    cdef public int player_index
    cdef public float expected_hand_strength
    cdef public bint folded
    cdef public int contributed_to_pot
    cdef public int tot_contributed_to_pot
    cdef public int prior_gains
    
    cpdef public void assign_position(self, str position, int player_index)
    cpdef public bint get_action(self, GameState game_state, int player_index)
    cpdef public bint take_action(self, GameState game_state, int player_index, object action)
    cpdef public list get_available_actions(self, GameState game_state)
    cpdef public str get_user_input(self, prompt)
    
    cpdef public void add_card(self, unsigned long long card)
    cpdef public void reset(self)
    cpdef public Player clone(self)
    cpdef public str hash(self, GameState game_state)