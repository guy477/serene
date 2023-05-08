from .player cimport Player
from .game_state cimport GameState

cdef class AIPlayer(Player):
    cpdef do_nothing(self)