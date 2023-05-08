#!python
#cython: language_level=3


cdef class Player:
    def __init__(self, int initial_chips):
        self.chips = initial_chips
        self.hand = 0
        self.folded = False
        self.contributed_to_pot = 0

    cpdef add_card(self, unsigned long long card):
        self.hand |= card

    cpdef reset(self):
        self.hand = 0
        self.folded = False
        self.contributed_to_pot = 0
