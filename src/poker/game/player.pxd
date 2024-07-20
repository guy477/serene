from .game_state cimport GameState
from .._utils._utils cimport *

cdef class Player:
    cdef public int chips
    cdef public list bet_sizing
    cdef public bint is_human
    cdef public unsigned long long hand
    cdef public str position
    cdef public int player_index
    cdef public float expected_hand_strength
    cdef public bint folded
    cdef public int contributed_to_pot
    cdef public int tot_contributed_to_pot
    cdef public int prior_gains
    
    cpdef void assign_position(self, str position, int player_index)
    cpdef object get_action(self, GameState game_state)
    cpdef bint take_action(self, GameState game_state, object action)
    cpdef list get_available_actions(self, GameState game_state)
    
    cpdef void add_card(self, unsigned long long card)
    cpdef void reset(self)
    cpdef Player clone(self)
    cpdef list hash(self, GameState game_state)