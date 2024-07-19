from .._utils._utils cimport *

from ..cfr.cfr cimport CFRTrainer
from ..core.local_manager cimport LocalManager
from .player cimport Player
from .game_state cimport GameState




cdef class PokerGame:
    cdef list suits
    cdef list values
    cdef logger
    cdef CFRTrainer strategy_trainer
    cdef LocalManager local_manager
    cdef GameState game_state
    cdef list profit_loss
    cdef dict position_pl
    cdef list players
    cdef list deck
    cpdef play_game(self, int num_hands=*)
    cpdef _play_round(self)
    cpdef get_action(self)
    cpdef skip_to_showdown(self)