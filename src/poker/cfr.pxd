from ._utils cimport *
from .game_state cimport GameState
from .player cimport Player

cdef class CFRTrainer:
    cdef int monte_carlo_depth
    cdef int prune_depth
    cdef double prune_probability_threshold
    cdef list suits
    cdef list values
    cdef int iterations
    cdef int realtime_iterations
    cdef int cfr_depth
    cdef int cfr_realtime_depth

    cdef int num_simulations

    cdef int num_players
    cdef int initial_chips
    cdef int small_blind
    cdef int big_blind

    cdef list bet_sizing
    cdef HashTable regret_sum
    cdef HashTable strategy_sum

    cpdef default_double(self)
    
    cdef progress_gamestate_to_showdown(self, GameState game_state, float epsilon = *)

    cpdef train(self, list positions_to_solve = *)

    cdef GameState fast_forward_gamestate(self, object hand, GameState game_state, list fast_forward_actions)

    cdef double[:] cfr_traverse(self, GameState game_state, double[:] probs, int depth, int max_depth, float epsilon = *)

    cdef double[:] calculate_utilities(self, GameState game_state, int player)

    cdef dict get_average_strategy(self, Player player, GameState game_state)
    
    cdef dict get_strategy(self, list available_actions, double[:] probs, GameState game_state, Player player, bint get_strategy)
    