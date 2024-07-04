from .player cimport Player
from .cfr cimport CFRTrainer
from .game_state cimport GameState
from ._utils cimport *

cdef class AIPlayer(Player):

    cdef public CFRTrainer strategy_trainer
    cpdef public AIPlayer clone(self)