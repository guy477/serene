from .player cimport Player

cdef class AIPlayer(Player):
    cpdef do_nothing(self)