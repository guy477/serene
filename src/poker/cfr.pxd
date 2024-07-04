from ._utils cimport *
from .game_state cimport GameState
from .ai_player cimport AIPlayer, Player
from .information_set cimport InformationSet

cdef class CFRTrainer:
    cdef public int monte_carlo_depth
    cdef public list suits
    cdef public list values
    cdef public int iterations
    cdef public int realtime_iterations
    cdef public int cfr_depth
    cdef public int cfr_realtime_depth

    cdef public int num_simulations

    cdef public int num_players
    cdef public int initial_chips
    cdef public int small_blind
    cdef public int big_blind

    cdef public list bet_sizing
    cdef public dict strategy_profiles
    cdef public dict regret_sum
    cdef public dict strategy_sum

    cpdef train(self, list positions_to_solve = *)

    cdef fast_forward_gamestate(self, object hand, GameState game_state, list fast_forward_actions)

    cpdef train_realtime(self, GameState game_state)

    cdef float[:] cfr_traverse(self, GameState game_state, float[:] probs, int depth, int max_depth, float epsilon = *)

    cdef float[:] calculate_utilities(self, GameState game_state, int player)

    cdef dict get_average_strategy(self, AIPlayer player, GameState game_state)
    
    cdef dict get_strategy(self, list available_actions, float[:] probs, GameState game_state, Player player)
    