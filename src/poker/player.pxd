from .game_state cimport GameState, display_game_state


cdef class Player:
    cdef public int chips
    cdef public unsigned long long hand
    cdef public bint folded
    cdef public int contributed_to_pot
    cdef public int tot_contributed_to_pot
    cdef public dict regret
    cdef public dict strategy_sum
    
    cpdef get_action(self, GameState game_state, int player_index)
    cpdef take_action(self, GameState game_state, int player_index, str action, int bet_size = *)

    cpdef get_available_actions(self, GameState game_state, int player_index)

    cpdef str get_user_input(self, prompt)

    cpdef get_strategy(self, list available_actions, float[:] probs, int current_player)

    cdef initialize_regret_strategy(self)

    cpdef add_card(self, unsigned long long card)
    cpdef reset(self)
    cpdef clone(self)