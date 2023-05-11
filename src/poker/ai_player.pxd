from .player cimport Player
from .cfr cimport CFRTrainer
from .game_state cimport GameState, display_game_state

cdef class AIPlayer(Player):
    cpdef public int cfr_depth
    cpdef public int cfr_realtime_depth
    cdef CFRTrainer strategy_trainer

    cpdef get_action(self, GameState game_state, int player_index)
    cpdef clone(self)