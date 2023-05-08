from .game_state cimport GameState, display_game_state


cdef class Player:
    cdef public int chips
    cdef public unsigned long long hand
    cdef public bint folded
    cdef public int contributed_to_pot
    
    cpdef get_action(self, GameState game_state, int player_index)
    cpdef str get_user_input(self, prompt)
    cpdef player_action(self, GameState game_state, int player_index, str action, int bet_amount=*)

    cpdef add_card(self, unsigned long long card)
    cpdef reset(self)