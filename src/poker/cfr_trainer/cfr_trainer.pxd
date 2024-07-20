
from .._utils._utils cimport *
from ..core.local_manager cimport LocalManager
from ..core.hash_table cimport HashTable
from ..game.game_state cimport GameState
from ..game.player cimport Player
from .cfr.cfr cimport CFR

cdef class CFRTrainer:
    cdef CFR cfr
    cdef int iterations
    cdef list suits
    cdef list values

    cdef int num_players
    cdef int initial_chips
    cdef int small_blind
    cdef int big_blind

    cdef list bet_sizing
    cpdef train(self, local_manager, list positions_to_solve = *, list hands = *, bint save_pickle = *)
    cdef fast_forward_gamestate(self, object hand, GameState game_state, list fast_forward_actions, LocalManager local_manager, int attempts = *)
    cpdef get_average_strategy_dump(self, fast_forward_actions, local_manager)
    cpdef generate_hands(self)
    cpdef double default_double(self)