from collections import defaultdict
import hashlib
import pickle

# Use cdef and cpdef for performance optimization
cpdef double default_double():
    return 0.0

cdef class HashTable:
    def __init__(self, shared_dict):
        self.table = shared_dict
        self.to_merge = {}
        self.to_prune = {}

    def __getitem__(self, key):
        cdef bytes hashed_key = abstract_key(key)
        return self.table[hashed_key]

    def __setitem__(self, key, value):
        cdef bytes hashed_key = abstract_key(key)
        if isinstance(value, tuple) and len(value) == 3:
            self.table[hashed_key] = value[0]
            if value[1]:
                self.to_prune[hashed_key] = value[1]
            if value[2]:
                self.to_merge[hashed_key] = value[2]
        else:
            raise ValueError("Value must be a tuple (actual_value, to_prune, to_merge)")

    def __contains__(self, key):
        cdef bytes hashed_key = abstract_key(key)
        return hashed_key in self.table

    def __len__(self):
        return len(self.table)

    def get(self, key, default=default_double):
        cdef bytes hashed_key = abstract_key(key)
        return self.table.get(hashed_key, default)

    def get_hashed(self, bytes hashed_key):
        return self.table[hashed_key]

    def get_set(self, key, default=None, prune=None, merge=None):
        cdef bytes hashed_key = abstract_key(key)
        
        if hashed_key not in self.table:
            self.set(hashed_key, default if default is not None else default_double)
        
        if prune:
            self.to_prune[hashed_key] = prune
        if merge:
            self.to_merge[hashed_key] = merge

        return self.table[hashed_key]

    def update(self, other):
        for key, value in other.items():
            self.table[key] = value

    def set(self, bytes hashed_key, value):
        self.table[hashed_key] = value

    def clear(self):
        self.table.clear()
        self.to_prune.clear()
        self.to_merge.clear()

    def items(self):
        return self.table.items()

    def dump(self, base_path, name):
        with open(f'{base_path}/{name}.pkl', 'wb') as f:
            pickle.dump(dict(self.table), f)

    def prune(self):
        for key in self.to_prune:
            if key not in self.to_merge:
                del self.table[key]
        self.to_prune.clear()



### NOTE: Load abstraction pickles at the global level..
### These are aggressive abstractions keyed by an already aggressive heuristic.
### TODO: Make this better. The resulting strategy is only as good as the abstraction. 
pickle_path_river = '../results/_post_flop_abstractions/river_abstraction_test.pkl'
pickle_path_turn = '../results/_post_flop_abstractions/turn_abstraction_test.pkl'
pickle_path_flop = '../results/_post_flop_abstractions/flop_abstraction_test.pkl'

cdef dict river_abstraction, turn_abstraction, flop_abstraction

with open(pickle_path_river, 'rb') as f:
    river_abstraction = dict(pickle.load(f))

with open(pickle_path_turn, 'rb') as f:
    turn_abstraction = dict(pickle.load(f))

with open(pickle_path_flop, 'rb') as f:
    flop_abstraction = dict(pickle.load(f))



cdef bytes abstract_key(list key):
    """Hash a tuple containing game_state information using SHA-256 and return the hexadecimal representation as bytes."""
    cdef:
        tuple key_hand_tuple = ulong_to_card_tuple(key[0])
        unsigned long long key_board_long = key[1]
        str key_position = key[2]
        int key_round_index = key[3]
        list key_available_actions = key[4]
        list key_action_space = key[5]
        float abstraction_harshness = 0.01
        object abstraction
        str new_key

    key_action_space = [
        [action for action in sublist if action[0] != "PUBLIC"]
        for sublist in key_action_space
    ]

    if key_round_index == 1: # flop
        abstraction = flop_abstraction[(abstract_hand(key_hand_tuple[0], key_hand_tuple[1]), handtype(key_board_long | key_hand_tuple[0] | key_hand_tuple[1], 5))] // abstraction_harshness
        new_key = str((abstraction, key_position, key_available_actions, key_action_space))
    elif key_round_index == 2: # turn
        abstraction = turn_abstraction[(abstract_hand(key_hand_tuple[0], key_hand_tuple[1]), handtype_partial(key_board_long | key_hand_tuple[0] | key_hand_tuple[1], 6))] // abstraction_harshness
        new_key = str((abstraction, key_position, key_available_actions, key_action_space))
    elif key_round_index == 3: # river
        abstraction = river_abstraction[(key_hand_tuple[0] | key_hand_tuple[1], handtype(key_board_long | key_hand_tuple[0] | key_hand_tuple[1], 7))] // abstraction_harshness
        new_key = str((abstraction, key_position, key_available_actions, key_action_space))
    else: # pre-flop
        new_key = str((abstract_hand(key_hand_tuple[0], key_hand_tuple[1]), key_position, key_available_actions, key_action_space))

    return hashlib.sha256(new_key.encode('utf-8')).digest()
