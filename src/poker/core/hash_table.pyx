from collections import defaultdict

import hashlib
import pickle

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

#########################

cpdef double default_double():
    ### 
    ### NOTE Hacky fix to de-reference the CFTrainer.default_double module.
    ### NOTE Global modules are not available to multiproceeses. But we can
    ### NOTE Package this one and send it in. 
    ###
    return 0.0

###

cdef class HashTable:
    """
    HashTable class implementing a hash table with SHA-256 hashed keys.
    
    This class provides an option to mark items for pruning and a method to
    remove all items marked for pruning.
    
    - `__init__(self)`: Initialize the hash table with an optional default value.
    - `__getitem__(self, key)`: Retrieve the value associated with the given key.
    - `__setitem__(self, key, value)`: Set the value for the given key. Value should be a tuple (actual_value, to_prune).
    - `__delitem__(self, key)`: Delete the item associated with the given key.
    - `__contains__(self, key)`: Check if the key exists in the table.
    - `get(self, key, default=defaultdict(float))`: Get the value for the key, ignoring the prune flag.
    - `update(self, other)`: Update the hash table with items from another dictionary.
    - `clear(self)`: Clear all items in the hash table.
    - `items(self)`: Get all items in the hash table.
    - `__len__(self)`: Get the number of items in the hash table.
    - `prune(self)`: Remove all items with `to_prune` set to `True`.
    """
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
            # value: tuple((tuple information_set, int pruned, int is_blueprint))
            self.table[hashed_key] = value[0]
            if value[1]:
                self.to_prune[hashed_key] = value[1]
            if value[2]:
                self.to_merge[hashed_key] = value[2]
        else:
            raise ValueError("Value must be a tuple (actual_value, to_prune)")


    def __contains__(self, key):
        cdef bytes hashed_key = abstract_key(key)
        return hashed_key in self.table

    def get(self, key, default = default_double):
        cdef bytes hashed_key = abstract_key(key)
        return self.table.get(hashed_key, default)

    def get_hashed(self, hashed_key):
        return self.table[hashed_key]

    def get_set(self, key, default=None, prune = None, merge = None):
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
            # We're assuming the other key is hashed already
            self.table[key] = value

    def set(self, hashed_key, value):
        self.table[hashed_key] = value

    def clear(self):
        self.table.clear()
        self.to_prune.clear()
        self.to_merge.clear()

    def items(self):
        for key, value in self.table.items():
            yield (key, value)

    def dump(self, base_path, name):
        self.table = convert_defaultdict(self.table)
        with open(f'{base_path}/{name}.pkl', 'wb') as f:
            pickle.dump(self.table, f)

    def __len__(self):
        return len(self.table)

    def prune(self):
        for key in self.to_prune:
            if key not in self.to_merge:
                del self.table[key]
                        
        self.to_prune.clear()

#########################

cdef bytes abstract_key(object key):
    """Hash a tuple continaing game_state information using SHA-256 and return the hexadecimal representation as bytes."""
    
    key_hand_tuple = ulong_to_card_tuple(key[0])
    key_board_long = key[1]
    key_position = key[2]
    key_round_index = key[3]
    key_available_actions = key[4]
    key_action_space = key[5]
    key_action_space = [
        [action for action in sublist if action[0] != "PUBLIC"]
        for sublist in key_action_space
    ]
    abstraction_harshness = .01

    if key_round_index == 1: # flop:
        abstraction = flop_abstraction[(abstract_hand(key_hand_tuple[0], key_hand_tuple[1]), handtype(key_board_long | key_hand_tuple[0] | key_hand_tuple[1], 5))] // abstraction_harshness
        new_key = str((abstraction, key_position, key_available_actions, key_action_space))
    elif key_round_index == 2: # turn:
        abstraction = turn_abstraction[(abstract_hand(key_hand_tuple[0], key_hand_tuple[1]), handtype_partial(key_board_long | key_hand_tuple[0] | key_hand_tuple[1], 6))] // abstraction_harshness
        new_key = str((abstraction, key_position, key_available_actions, key_action_space))
    elif key_round_index == 3: # river:
        abstraction = river_abstraction[(key_hand_tuple[0] | key_hand_tuple[1], handtype(key_board_long | key_hand_tuple[0] | key_hand_tuple[1], 7))] // abstraction_harshness
        new_key = str((abstraction, key_position, key_available_actions, key_action_space))
    else:
        new_key = str((abstract_hand(key_hand_tuple[0], key_hand_tuple[1]), key_position, key_available_actions, key_action_space))
           
    return hashlib.sha256(new_key.encode('utf-8')).digest()

#########################

# Make the defaultdict environment agnostic
def convert_defaultdict(d):
    if isinstance(d, defaultdict):
        # Check if the defaultdict uses a bound method `default_double` and replace it with float
        if d.default_factory and d.default_factory.__name__ == 'default_double':
            new_d = defaultdict(default_double, {k: convert_defaultdict(v) for k, v in d.items()})
        else:
            new_d = defaultdict(d.default_factory, {k: convert_defaultdict(v) for k, v in d.items()})
        return new_d
    elif isinstance(d, dict):
        # Recursively convert each value in the dict
        return {k: convert_defaultdict(v) for k, v in d.items()}
    else:
        return d

