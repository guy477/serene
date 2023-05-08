from .game_state cimport GameState
from .player cimport Player
from .information_set cimport InformationSet

cdef class CFRTrainer:
    cdef public int iterations
    cdef public int num_players
    cdef public int initial_chips
    cdef public int small_blind
    cdef public int big_blind
    cdef public dict strategy_profiles
    cdef public dict regret_sum
    cdef public dict strategy_sum

    cpdef train(self)

    cpdef double traverse_game_tree(self, GameState game_state, int player_index, list reach_probabilities)

    cpdef list get_strategy(self, str information_set, int player_index, list available_actions)

    cpdef str get_best_action(self, GameState game_state, int player_index)
