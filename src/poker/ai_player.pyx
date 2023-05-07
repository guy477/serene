#!python
#cython: language_level=3

cdef class AIPlayer(Player):
    # AI player-specific methods will be added here
    cpdef do_nothing(self):
        return
