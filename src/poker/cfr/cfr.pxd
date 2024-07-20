from .._utils._utils cimport *
from ..core.local_manager cimport LocalManager
from ..core.hash_table cimport HashTable
from ..game.game_state cimport GameState
from ..game.player cimport Player

cdef class CFRTrainer:
    cdef int prune_depth
    cdef double prune_probability_threshold
    cdef list suits
    cdef list values
    cdef int iterations
    cdef int cfr_depth

    cdef int num_players
    cdef int initial_chips
    cdef int small_blind
    cdef int big_blind

    cdef list bet_sizing
    
    cdef LocalManager local_manager

    cpdef train(self, local_manager, list positions_to_solve = *, list hands = *, bint save_pickle=*)

    cdef fast_forward_gamestate(self, object hand, GameState game_state, list fast_forward_actions, LocalManager local_manager, int attempts = *)
    
    cdef progress_gamestate_to_showdown(self, GameState game_state)

    cdef double[:] cfr_traverse(self, GameState game_state, double[:] probs, int depth, int max_depth, LocalManager local_manager)

    cdef double[:] calculate_utilities(self, GameState game_state, int player)

    cdef dict get_strategy(self, list available_actions, double[:] probs, GameState game_state, Player player, LocalManager local_manager)
    
    cpdef dict get_average_strategy(self, Player player, GameState game_state, LocalManager local_manager)

    cpdef get_average_strategy_dump(self, fast_forward_actions, local_manager)

    cpdef double default_double(self)

    cpdef generate_hands(self)
