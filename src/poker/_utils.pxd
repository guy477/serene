import numpy
cimport numpy
cimport cython

cdef public list SUITS = ['C', 'D', 'H', 'S']
cdef public list VALUES = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']

cdef public dict SUITS_INDEX = {'C': 0, 'D': 1, 'H': 2, 'S': 3}
cdef public dict VALUES_INDEX = {'2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6, '9': 7, 'T': 8, 'J': 9, 'Q': 10, 'K': 11, 'A': 12}


# 'Global' functions
cdef unsigned long long card_to_int(str suit, str value)
cdef str int_to_card(unsigned long long card)
cpdef unsigned long long card_str_to_int(str card_str)
cdef tuple card_tuple_to_str_tuple(tuple cards)
cdef tuple ulong_to_card_tuple(unsigned long long hand)
cdef str format_hand(unsigned long long hand)
cdef display_game_state(object game_state, int player_index)
cdef unsigned int cy_evaluate(unsigned long long cards, unsigned int num_cards) nogil
cdef unsigned int cy_evaluate_handtype(unsigned long long cards, unsigned int num_cards) nogil
cpdef handtype(unsigned long long hand_board, unsigned int num_cards)
cpdef handtype_partial(unsigned long long hand_board, unsigned int num_cards)
cpdef cy_evaluate_cpp(cards, num_cards)
cpdef list build_fast_forward_actions(list action_space)
cpdef object select_random_action(average_strategy)
cdef str abstract_hand(unsigned long long card1, unsigned long long card2)
cpdef double default_double()

cdef class HashTable:
    cdef public object table
    cdef public dict to_merge
    cdef public dict to_prune

cdef class LocalManager:
    cdef public HashTable regret_sum
    cdef public HashTable strategy_sum

cpdef dynamic_merge_dicts(local_manager_table, global_accumulator)

cdef class Deck:
    cdef list suits
    cdef list values
    cdef list deck

    cdef list to_list(self)
    cdef list create_deck(self)
    cdef void remove(self, unsigned long long card)
    cdef void add(self, unsigned long long card)
    cdef void fisher_yates_shuffle(self)
    cdef unsigned long long pop(self)
    cdef Deck clone(self)
    cdef void reset(self)
    