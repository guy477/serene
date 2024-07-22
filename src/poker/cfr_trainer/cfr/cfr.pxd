from ..._utils._utils cimport *
from ...core.local_manager cimport LocalManager
from ...core.hash_table cimport HashTable
from ...game.game_state cimport GameState
from ...game.player cimport Player

cdef class CFR:
    cdef int prune_depth
    cdef double prune_probability_threshold
    cdef int cfr_depth
    
    cdef progress_gamestate_to_showdown(self, GameState game_state, LocalManager local_manager)

    cdef double[:] cfr_traverse(self, GameState game_state, double[:] probs, int depth, LocalManager local_manager)

    cdef double[:] calculate_utilities(self, GameState game_state, int player)

    cdef dict get_strategy(self, list available_actions, double[:] probs, GameState game_state, Player player, LocalManager local_manager)
    
    cpdef dict get_average_strategy(self, Player player, GameState game_state, LocalManager local_manager)

    cpdef double default_double(self)
