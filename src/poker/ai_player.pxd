from .player cimport Player
from .cfr cimport CFRTrainer
from .game_state cimport GameState
from ._utils cimport *

cdef class AIPlayer(Player):
    cpdef bint get_action(self, GameState game_state, int player_index, CFRTrainer strategy_trainer = *)
    cpdef public AIPlayer clone(self)