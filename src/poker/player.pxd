
cdef class Player:
    cdef public int chips
    cdef public unsigned long long hand
    cdef public bint folded
    cdef public int contributed_to_pot
    cpdef add_card(self, unsigned long long card)
    cpdef reset(self)