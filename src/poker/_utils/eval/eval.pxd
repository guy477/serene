from .._utils cimport *

cpdef handtype(unsigned long long hand_board, unsigned int num_cards)
cpdef handtype_partial(unsigned long long hand_board, unsigned int num_cards)

cdef unsigned int cy_evaluate(unsigned long long cards, unsigned int num_cards) nogil
cdef unsigned int cy_evaluate_handtype(unsigned long long cards, unsigned int num_cards) nogil
cpdef cy_evaluate_cpp(cards, num_cards)