cimport numpy
cimport cython

cdef public list SUITS = ['C', 'D', 'H', 'S']
cdef public list VALUES = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']

cdef public dict SUITS_INDEX = {'C': 0, 'D': 1, 'H': 2, 'S': 3}
cdef public dict VALUES_INDEX = {'2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6, '9': 7, 'T': 8, 'J': 9, 'Q': 10, 'K': 11, 'A': 12}

## GAMESTATE/HASH
cpdef dynamic_merge_dicts(local_manager_table, global_accumulator)
cpdef list build_fast_forward_actions(list action_space)
cpdef object select_random_action(average_strategy)

## DECK/CARDS
cdef unsigned long long card_to_int(str suit, str value)
cdef str int_to_card(unsigned long long card)
cpdef unsigned long long card_str_to_int(str card_str)
cdef tuple card_tuple_to_str_tuple(tuple cards)
cdef tuple ulong_to_card_tuple(unsigned long long hand)
cdef str format_hand(unsigned long long hand)
cdef display_game_state(object game_state, int player_index)
cdef str abstract_hand(unsigned long long card1, unsigned long long card2)




