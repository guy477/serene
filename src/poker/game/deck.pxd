from .._utils._utils cimport *

cdef class Deck:
    cdef list suits
    cdef list values
    cdef list deck

    cdef list to_list(self)
    cdef void create_deck(self)
    cdef void remove(self, unsigned long long card)
    cdef void add(self, unsigned long long card)
    cdef void shuffle(self)
    cdef unsigned long long pop(self)
    cdef Deck clone(self)
    cdef void reset(self)
    