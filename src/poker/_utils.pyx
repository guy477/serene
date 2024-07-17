import os
import pickle
import random
from collections import defaultdict

cdef public list SUITS = ['C', 'D', 'H', 'S']
cdef public list VALUES = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']

cdef public dict SUITS_INDEX = {'C': 0, 'D': 1, 'H': 2, 'S': 3}
cdef public dict VALUES_INDEX = {'2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6, '9': 7, 'T': 8, 'J': 9, 'Q': 10, 'K': 11, 'A': 12}



### NOTE: Load abstraction pickles at the global level..
### These are aggressive abstractions keyed by an already aggressive heuristic.
### TODO: Make this better. The resulting strategy is only as good as the abstraction. 
pickle_path_river = 'dat/pickles/post_flop_abstractions/river_abstraction_test.pkl'
pickle_path_turn = 'dat/pickles/post_flop_abstractions/turn_abstraction_test.pkl'
pickle_path_flop = 'dat/pickles/post_flop_abstractions/flop_abstraction_test.pkl'

cdef dict river_abstraction, turn_abstraction, flop_abstraction

with open(pickle_path_river, 'rb') as f:
    river_abstraction = dict(pickle.load(f))

with open(pickle_path_turn, 'rb') as f:
    turn_abstraction = dict(pickle.load(f))

with open(pickle_path_flop, 'rb') as f:
    flop_abstraction = dict(pickle.load(f))



import numpy
cimport numpy
cimport cython

from tqdm import tqdm

import hashlib


cdef class LocalManager:
    def __init__(self, pkl_path_regret = 'regret_sum.pkl', pkl_path_strategy = 'strategy_sum.pkl'):
        self.load(pkl_path_regret, pkl_path_strategy)
    
    def get_regret_sum(self):
        return self.regret_sum
    
    def get_strategy_sum(self):
        return self.strategy_sum

    def set_regret_sum_table(self, regret_sum_other):
        self.regret_sum.table = regret_sum_other

    def set_strategy_sum_table(self, strategy_sum_other):
        self.strategy_sum.table = strategy_sum_other
    
    def merge_dicts(self, hash_table, shared_dict):
        for key, value in shared_dict.items():
            hash_table.set(key, value)

    def save(self, regret_sum_path, strategy_sum_path):
        print(f"Saving regret sum to {regret_sum_path} and strategy sum to {strategy_sum_path}")
        with open(regret_sum_path, 'wb') as f:
            pickle.dump(convert_defaultdict(self.regret_sum.table), f)

        with open(strategy_sum_path, 'wb') as f:
            pickle.dump(convert_defaultdict(self.strategy_sum.table), f)
    
    def load(self, regret_sum_path, strategy_sum_path):
        print(f"Loading regret sum from {regret_sum_path} and strategy sum from {strategy_sum_path}")
        try:
            with open(regret_sum_path, 'rb') as f:
                self.regret_sum = HashTable(pickle.load(f))
        except Exception as e:
            print(f"Error loading regret sum: {e}")
            self.regret_sum = HashTable({})

        try:
            with open(strategy_sum_path, 'rb') as f:
                self.strategy_sum = HashTable(pickle.load(f))
        except Exception as e:
            print(f"Error loading regret sum: {e}")
            self.strategy_sum = HashTable({})


cpdef double default_double():
    ### 
    ### NOTE Hacky fix to de-reference the CFTrainer.default_double modeul.
    ### NOTE Global modules are not available to multiproceeses. But we can
    ### NOTE Package this one and send it in. For simplicity, CFTrainer will
    ### NOTE keep its own default_double and convert before offloading to disk.
    ###
    return 0.0

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


cdef bytes hash_key_sha256(object key):
    """Hash a tuple continaing game_state information using SHA-256 and return the hexadecimal representation as bytes."""
    
    key_hand_list = key[0]
    key_board_long = key[1]
    key_position = key[2]
    key_round_index = key[3]
    key_available_actions = key[4]
    key_betting_history = key[5]

    abstraction_harshness = .01

    if key_round_index == 1: # flop:
        abstraction = flop_abstraction[(abstract_hand(key_hand_list[0], key_hand_list[1]), handtype(key_board_long | key_hand_list[0] | key_hand_list[1], 5))] // abstraction_harshness
        new_key = str((abstraction, key_position, key_available_actions, key_betting_history))
    elif key_round_index == 2: # turn:
        abstraction = turn_abstraction[(abstract_hand(key_hand_list[0], key_hand_list[1]), handtype_partial(key_board_long | key_hand_list[0] | key_hand_list[1], 6))] // abstraction_harshness
        new_key = str((abstraction, key_position, key_available_actions, key_betting_history))
    elif key_round_index == 3: # river:
        abstraction = river_abstraction[(key_hand_list[0] | key_hand_list[1], handtype(key_board_long | key_hand_list[0] | key_hand_list[1], 7))] // abstraction_harshness
        new_key = str((abstraction, key_position, key_available_actions, key_betting_history))
    else:
        new_key = str((abstract_hand(key_hand_list[0], key_hand_list[1]), key_position, key_available_actions, key_betting_history))

    return hashlib.sha256(new_key.encode('utf-8')).digest()

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
        cdef bytes hashed_key = hash_key_sha256(key)
        return self.table[hashed_key]

    def __setitem__(self, key, value):
        cdef bytes hashed_key = hash_key_sha256(key)
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
        cdef bytes hashed_key = hash_key_sha256(key)
        return hashed_key in self.table

    def get(self, key, default = default_double):
        cdef bytes hashed_key = hash_key_sha256(key)
        return self.table.get(hashed_key, default)

    def get_hashed(self, hashed_key):
        return self.table[hashed_key]

    def get_set(self, key, default=None, prune = True, merge = False):
        cdef bytes hashed_key = hash_key_sha256(key)
        
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

    def __len__(self):
        return len(self.table)

    def prune(self):
        for key in self.to_prune:
            del self.table[key]
            if key in self.to_merge:
                del self.to_merge[key]
        self.to_prune.clear()

cpdef dynamic_merge_dicts(local_manager_table, global_accumulator):
    
    for player_hash_local in tqdm(local_manager_table.to_merge):
        inner_dict = local_manager_table.get_hashed(player_hash_local)
        global_accumulator[player_hash_local] = inner_dict
        # Look at this schizo rambling... look the more up-to-date values better reflect a more optimized, less random, strategy. Just take the new values and keep it simple.
        """
        if player_hash_local in global_accumulator:
            existing_inner_dict = global_accumulator[player_hash_local]
            
            ##      THIS SHOULD BE THE SUM, NOT AVERAGE.
            ##      I USE AVERAGE BECAUSE WHEN WE TRAIN FOR NODE N, WE ACCUMULATE FOR NODE N+1 - OF WHICH, THE MAJORITY OF STEPS TAKEN FROM N ARE RANDOM - CAUSING THE ACCUMULATED VALUES TO REFLECT
            ##          A RANDOM OP PONENT. SINCE THESE ACCUMULATIONS HAPPEN DEEPER IN THE ACTION TREE, WE END UP ACCUMULATING MORE AT N+1 COMPARED TO N (N CAN LEAD TO N+1 MULTIPLE TIMES IN THE SAME ITERATION). 
            ##
            ##      NOTE: IDEALLY, average(everything) == ((((everything[0]+everything[1])/2 + everything[3])/2 + everything[4])...) 
            ##              The assumption here is that everything[n] accumulates the same number of elements as everything[n-1]
            ##              This does not hold true below.
            ##                  INVESTIGATE HOW TO PROPERLY ACCUMULATE STRATEGIES... SHOULD WE ONLY ACCUMULATE FOR NODE N WHILE BUILDING A BLUEPRINT STRATEGY?
            ##                  Arguments can be made in favor of this logic below (balances exploring with exploitation)... Verification is needed.
            ##
            for inner_key, inner_value in inner_dict.items():
                if inner_key in existing_inner_dict:
                    existing_inner_dict[inner_key] = (existing_inner_dict[inner_key] + inner_value)/2
                else:
                    existing_inner_dict[inner_key] = inner_value
            
            # This value is part of the existing blueprint and we've merged the new results with the existing results
            global_accumulator[player_hash_local] = existing_inner_dict
        else:
            # If key does not exist, simply add the new entry
            global_accumulator[player_hash_local] = inner_dict
        """
        # else:
        #   the node being considered is part of a precomputed strategy and *shouldn't* have updates
cdef class Deck:

    def __init__(self, list suits, list values):
        self.suits = suits
        self.values = values

    cdef list create_deck(self):
        cdef int size = len(self.suits) * len(self.values)
        cdef list deck = [0] * size
        cdef int index = 0
        cdef int suit_index, value_index
        cdef str suit, value
        for suit in self.suits:
            suit_index = SUITS_INDEX[suit]
            for value in self.values:
                value_index = VALUES_INDEX[value]
                deck[index] = card_to_int(suit, value)
                index += 1
        return deck

    cdef void fisher_yates_shuffle(self):
        numpy.random.shuffle(self.deck)
        # pass

    cdef unsigned long long pop(self):
        # Remove from the beginning
        return self.deck.pop(0)

    cdef void remove(self, unsigned long long card):
        # Remove the given card
        self.deck.remove(card)

    cdef void add(self, unsigned long long card):
        # Add to the end
        self.deck.append(card)

    cdef list to_list(self):
        # Deck is a list; but if we were to optimize deck to another struct...
        return self.deck

    cdef Deck clone(self):
        cdef Deck new_deck = Deck(self.suits, self.values)
        new_deck.deck = self.deck[:]
        return new_deck

    cdef void reset(self):
        self.deck = self.create_deck()
        self.fisher_yates_shuffle()


cpdef list build_fast_forward_actions(list betting_history):
    cdef list fast_forward_actions = betting_history[0][2:] + betting_history[1] + betting_history[2] + betting_history[3]
    
    # Sort of confusing function.. hence why it's a utility.
    return [pos_w_action[1] for pos_w_action in fast_forward_actions]


cdef str abstract_hand(unsigned long long card1, unsigned long long card2):
    cdef str card1_str = int_to_card(card1)
    cdef str card2_str = int_to_card(card2)

    # Temporary variables for the card values
    cdef str card1_val = card1_str[0]
    cdef str card2_val = card2_str[0]

    # Now use the temporary variables in your comparison
    cdef str high_card = card1_val if VALUES_INDEX[card1_val] > VALUES_INDEX[card2_val] else card2_val
    cdef str low_card = card1_val if VALUES_INDEX[card1_val] < VALUES_INDEX[card2_val] else card2_val
    cdef str suited = 's' if card1_str[1] == card2_str[1] else 'o'
    
    return high_card + low_card + suited

cpdef object select_action(average_strategy):
    actions = list(average_strategy.keys())
    probabilities = list(average_strategy.values())
    selected_action = random.choices(actions, probabilities)[0]
    return selected_action

cdef unsigned long long card_to_int(str suit, str value):
    cdef unsigned long long one = 1
    cdef int suit_index = SUITS_INDEX[suit]
    cdef int value_index = VALUES_INDEX[value]
    cdef int bit_position = suit_index * 13 + value_index
    return one << bit_position

cdef str int_to_card(unsigned long long card):
    cdef int bit_position = -1
    while card > 0:
        card >>= 1
        bit_position += 1
    cdef int suit_index = bit_position // 13
    cdef int value_index = bit_position % 13
    return f'{VALUES[value_index]}{SUITS[suit_index]}'


cpdef unsigned long long card_str_to_int(str card_str):
    return card_to_int(card_str[1], card_str[0])

cdef public list create_deck():
    cdef list deck = [card_to_int(suit, value) for suit in SUITS for value in VALUES]
    return deck

cdef list hand_to_cards(unsigned long long hand):
    cdef list cards = [card for card in create_deck() if card & hand]
    return cards

cdef str format_hand(unsigned long long hand):
    cdef list cards = [int_to_card(card) for card in create_deck() if card & hand]
    return " ".join(cards)


####################################################################################################
####################################################################################################
################################## DISPLAY CURRENT GAME STATUS #####################################
####################################################################################################
####################################################################################################
import os

# Helper function to clear the console
def clear_console():
    input('press enter to clear')
    os.system('clear')

# Helper function to format player contributions
def format_contributions(player, game_state):
    contributions = player.tot_contributed_to_pot
    # if player.position == 'SB':
    #     contributions += game_state.small_blind
    # elif player.position == 'BB':
    #     contributions += game_state.big_blind
    return f"{contributions}".ljust(5)

# Helper function to format player status
def format_status(player, current_player, player_index):
    status = 'folded' if player.folded else 'active'
    highlight = '   <---' if player.position == current_player.position and player == player_index else ''
    return f"{status} --- {player.chips}{' ' * (6 - len(str(player.chips)))} --- {player.prior_gains}{highlight}"

# Helper function to display player information
def display_player_info(player, game_state, current_player, player_index):
    contributions = format_contributions(player, game_state)
    status = format_status(player, current_player, player_index)
    return f"({player.position}){' ' * (8 - len('_' + player.position + '_'))}: {format_hand(player.hand)}   --- {contributions}   --- {status}"

# Helper function to display the game state header
def display_header(game_state, current_player):
    folded = {plr.position for plr in game_state.players if plr.folded}
    last_move = next((item for sublist in reversed(game_state.betting_history) for item in reversed(sublist) if item is not None and (item[0] not in folded or item[1][0] == 'fold')), None)
    
    header = (
        f"______________________________________________________________________________\n"
        f"({current_player.position}): {format_hand(current_player.hand)} --- {'folded' if current_player.folded else 'active'}\n"
        f"Last move: {last_move}\n"
        f"Board: {format_hand(game_state.board)}\n"
        f"Pot: {game_state.pot}\n"
        f"Chips: {current_player.chips}\n"
        f"______________________________________________________________________________\n"
    )
    return header

# Helper function to display the betting rounds
def display_betting_rounds(game_state):
    betting_rounds = (
        f'\n          {" " * game_state.cur_round_index * 20}|\n'
        f'          {" " * game_state.cur_round_index * 20}V\n'
        "        PREFLOP     ---      FLOP      ---      TURN      ---     RIVER"
    )
    return betting_rounds

# Helper function to display actions dictionary
def display_actions(actions_dict, rounds):
    max_len = max(len(pos) for pos in actions_dict.keys()) + 2
    actions_display = ""
    for pos, actions in actions_dict.items():
        actions_display += f"{pos:<{max_len}} {actions['Preflop']:<18} {actions['Flop']:<18} {actions['Turn']:<18} {actions['River']:<18}\n"
    return actions_display

# Helper function to generate actions dictionary
def generate_actions_dict(game_state, folded):
    rounds = ['Preflop', 'Flop', 'Turn', 'River']
    actions_dict = {player.position: {round: ' ' * 18 for round in rounds} for player in game_state.players}

    for round_idx, round_actions in enumerate(game_state.betting_history):
        if round_idx > game_state.cur_round_index:
            break
        for player in game_state.players:
            player_pos = player.position
            player_action = next((action for pos, action in reversed(round_actions) if pos == player_pos), ('', ''))
            if player_pos in folded:
                player_action = ('fold', 0)
            actions_dict[player_pos][rounds[round_idx]] = f"{player_action[0][:5].ljust(7)} ({str(player_action[1])[:5]})"
    
    return actions_dict, rounds

# Main function to display the game state
cdef display_game_state(object game_state, int player_index):
    clear_console()
    
    current_player = game_state.get_current_player()
    folded = {plr.position for plr in game_state.players if plr.folded}

    print(display_header(game_state, current_player))
    
    print(f"\nPOS      CARDS    POT CONTRIBS    STATUS     STACK     PRIOR GAINS")
    for i, player in enumerate(game_state.players):
        print(display_player_info(player, game_state, current_player, player_index))
    print(f"______________________________________________________________________________")
    
    print(display_betting_rounds(game_state))
    
    actions_dict, rounds = generate_actions_dict(game_state, folded)
    print(display_actions(actions_dict, rounds))



####################################################################################################
####################################################################################################
####################################################################################################
# The below code is taken from eval7 - https://pypi.org/project/eval7/
####################################################################################################
####################################################################################################
####################################################################################################


ctypedef numpy.uint8_t uint8
ctypedef numpy.uint16_t uint16
ctypedef numpy.int16_t int16
ctypedef numpy.int64_t int64
ctypedef numpy.npy_bool boolean


cdef extern from "arrays.h":
    unsigned short N_BITS_TABLE[8192]
    unsigned short STRAIGHT_TABLE[8192]
    unsigned int TOP_FIVE_CARDS_TABLE[8192]
    unsigned short TOP_CARD_TABLE[8192]

cdef int CLUB_OFFSET = 0
cdef int DIAMOND_OFFSET = 13
cdef int HEART_OFFSET = 26
cdef int SPADE_OFFSET = 39

cdef int HANDTYPE_SHIFT = 24 
cdef int TOP_CARD_SHIFT = 16 
cdef int SECOND_CARD_SHIFT = 12 
cdef int THIRD_CARD_SHIFT = 8 
cdef int CARD_WIDTH = 4 
cdef unsigned int TOP_CARD_MASK = 0x000F0000 
cdef unsigned int SECOND_CARD_MASK = 0x0000F000 
cdef unsigned int FIFTH_CARD_MASK = 0x0000000F 

cdef unsigned int HANDTYPE_VALUE_STRAIGHTFLUSH = ((<unsigned int>8) << HANDTYPE_SHIFT)
cdef unsigned int HANDTYPE_VALUE_FOUR_OF_A_KIND = ((<unsigned int>7) << HANDTYPE_SHIFT)
cdef unsigned int HANDTYPE_VALUE_FULLHOUSE = ((<unsigned int>6) << HANDTYPE_SHIFT)
cdef unsigned int HANDTYPE_VALUE_FLUSH = ((<unsigned int>5) << HANDTYPE_SHIFT)
cdef unsigned int HANDTYPE_VALUE_STRAIGHT = ((<unsigned int>4) << HANDTYPE_SHIFT)
cdef unsigned int HANDTYPE_VALUE_TRIPS = ((<unsigned int>3) << HANDTYPE_SHIFT)
cdef unsigned int HANDTYPE_VALUE_TWOPAIR = ((<unsigned int>2) << HANDTYPE_SHIFT)
cdef unsigned int HANDTYPE_VALUE_PAIR = ((<unsigned int>1) << HANDTYPE_SHIFT)
cdef unsigned int HANDTYPE_VALUE_HIGHCARD = ((<unsigned int>0) << HANDTYPE_SHIFT)


cdef unsigned int cy_evaluate(unsigned long long cards, unsigned int num_cards) nogil:
    """
    7-card evaluation function based on Keith Rule's port of PokerEval.
    Pure Python: 20000 calls in 0.176 seconds (113636 calls/sec)
    Cython: 20000 calls in 0.044 seconds (454545 calls/sec)
    """
    cdef unsigned int retval = 0, four_mask, three_mask, two_mask
    
    cdef unsigned int sc = <unsigned int>((cards >> (CLUB_OFFSET)) & 0x1fffUL)
    cdef unsigned int sd = <unsigned int>((cards >> (DIAMOND_OFFSET)) & 0x1fffUL)
    cdef unsigned int sh = <unsigned int>((cards >> (HEART_OFFSET)) & 0x1fffUL)
    cdef unsigned int ss = <unsigned int>((cards >> (SPADE_OFFSET)) & 0x1fffUL)
    
    cdef unsigned int ranks = sc | sd | sh | ss
    cdef unsigned int n_ranks = N_BITS_TABLE[ranks]
    cdef unsigned int n_dups = <unsigned int>(num_cards - n_ranks)
    
    cdef unsigned int st, t, kickers, second, tc, top
    
    if n_ranks >= 5:
        if N_BITS_TABLE[ss] >= 5:
            if STRAIGHT_TABLE[ss] != 0:
                return HANDTYPE_VALUE_STRAIGHTFLUSH + <unsigned int>(STRAIGHT_TABLE[ss] << TOP_CARD_SHIFT)
            else:
                retval = HANDTYPE_VALUE_FLUSH + TOP_FIVE_CARDS_TABLE[ss]
        elif N_BITS_TABLE[sc] >= 5:
            if STRAIGHT_TABLE[sc] != 0:
                return HANDTYPE_VALUE_STRAIGHTFLUSH + <unsigned int>(STRAIGHT_TABLE[sc] << TOP_CARD_SHIFT)
            else:
                retval = HANDTYPE_VALUE_FLUSH + TOP_FIVE_CARDS_TABLE[sc]
        elif N_BITS_TABLE[sd] >= 5:
            if STRAIGHT_TABLE[sd] != 0:
                return HANDTYPE_VALUE_STRAIGHTFLUSH + <unsigned int>(STRAIGHT_TABLE[sd] << TOP_CARD_SHIFT)
            else:
                retval = HANDTYPE_VALUE_FLUSH + TOP_FIVE_CARDS_TABLE[sd]
        elif N_BITS_TABLE[sh] >= 5:
            if STRAIGHT_TABLE[sh] != 0:
                return HANDTYPE_VALUE_STRAIGHTFLUSH + <unsigned int>(STRAIGHT_TABLE[sh] << TOP_CARD_SHIFT)
            else:
                retval = HANDTYPE_VALUE_FLUSH + TOP_FIVE_CARDS_TABLE[sh]
        else:
            st = STRAIGHT_TABLE[ranks]
            if st != 0:
                retval = HANDTYPE_VALUE_STRAIGHT + (st << TOP_CARD_SHIFT)

        if retval != 0 and n_dups < 3:
            return retval

    if n_dups == 0:
        return HANDTYPE_VALUE_HIGHCARD + TOP_FIVE_CARDS_TABLE[ranks]
    elif n_dups == 1:
        two_mask = ranks ^ (sc ^ sd ^ sh ^ ss)
        retval = <unsigned int>(HANDTYPE_VALUE_PAIR + (TOP_CARD_TABLE[two_mask] << TOP_CARD_SHIFT))
        t = ranks ^ two_mask
        kickers = (TOP_FIVE_CARDS_TABLE[t] >> CARD_WIDTH) & ~FIFTH_CARD_MASK
        retval += kickers
        return retval
    elif n_dups == 2:
        two_mask = ranks ^ (sc ^ sd ^ sh ^ ss)
        if two_mask != 0:
            t = ranks ^ two_mask
            retval = <unsigned int>(HANDTYPE_VALUE_TWOPAIR
                + (TOP_FIVE_CARDS_TABLE[two_mask]
                & (TOP_CARD_MASK | SECOND_CARD_MASK))
                + (TOP_CARD_TABLE[t] << THIRD_CARD_SHIFT))
            return retval
        else:
            three_mask = ((sc & sd) | (sh & ss)) & ((sc & sh) | (sd & ss))
            retval = <unsigned int>(HANDTYPE_VALUE_TRIPS + (TOP_CARD_TABLE[three_mask] << TOP_CARD_SHIFT))
            t = ranks ^ three_mask
            second = TOP_CARD_TABLE[t]
            retval += (second << SECOND_CARD_SHIFT)
            t ^= (1U << <int>second)
            retval += <unsigned int>(TOP_CARD_TABLE[t] << THIRD_CARD_SHIFT)
            return retval
    else:
        four_mask = sh & sd & sc & ss
        if four_mask != 0:
            tc = TOP_CARD_TABLE[four_mask]
            retval = <unsigned int>(HANDTYPE_VALUE_FOUR_OF_A_KIND
                + (tc << TOP_CARD_SHIFT)
                + ((TOP_CARD_TABLE[ranks ^ (1U << <int>tc)]) << SECOND_CARD_SHIFT))
            return retval
        two_mask = ranks ^ (sc ^ sd ^ sh ^ ss)
        if N_BITS_TABLE[two_mask] != n_dups:
            three_mask = ((sc & sd) | (sh & ss)) & ((sc & sh) | (sd & ss))
            retval = HANDTYPE_VALUE_FULLHOUSE
            tc = TOP_CARD_TABLE[three_mask]
            retval += (tc << TOP_CARD_SHIFT)
            t = (two_mask | three_mask) ^ (1U << <int>tc)
            retval += <unsigned int>(TOP_CARD_TABLE[t] << SECOND_CARD_SHIFT)
            return retval
        if retval != 0:
            return retval
        else:
            retval = HANDTYPE_VALUE_TWOPAIR
            top = TOP_CARD_TABLE[two_mask]
            retval += (top << TOP_CARD_SHIFT)
            second = TOP_CARD_TABLE[two_mask ^ (1 << <int>top)]
            retval += (second << SECOND_CARD_SHIFT)
            retval += <unsigned int>((TOP_CARD_TABLE[ranks ^ (1U << <int>top) ^ (1 << <int>second)]) << THIRD_CARD_SHIFT)
            return retval


cdef unsigned int cy_evaluate_handtype(unsigned long long cards, unsigned int num_cards) nogil:
    """
    This function is a duplicate of the one above. Here I pack more information into the return value.
    This enables the handtype funciton to more accurately describe a hand.
    The change likely results in different hand-rankings on the margins since this extra information changes the returned value.
    Since this change is untested:
                    
                     *****THIS FUNCTION IS TO BE USED ONLY BY "_utils.handtype"*****
    """
    cdef unsigned int retval = 0, four_mask, three_mask, two_mask
    
    cdef unsigned int sc = <unsigned int>((cards >> (CLUB_OFFSET)) & 0x1fffUL)
    cdef unsigned int sd = <unsigned int>((cards >> (DIAMOND_OFFSET)) & 0x1fffUL)
    cdef unsigned int sh = <unsigned int>((cards >> (HEART_OFFSET)) & 0x1fffUL)
    cdef unsigned int ss = <unsigned int>((cards >> (SPADE_OFFSET)) & 0x1fffUL)
    
    cdef unsigned int ranks = sc | sd | sh | ss
    cdef unsigned int n_ranks = N_BITS_TABLE[ranks]
    cdef unsigned int n_dups = <unsigned int>(num_cards - n_ranks)
    
    cdef unsigned int st, t, kickers, second, tc, top
    
    if n_ranks >= 5:
        if N_BITS_TABLE[ss] >= 5:
            if STRAIGHT_TABLE[ss] != 0:
                return HANDTYPE_VALUE_STRAIGHTFLUSH + <unsigned int>(STRAIGHT_TABLE[ss] << TOP_CARD_SHIFT)
            else:
                retval = HANDTYPE_VALUE_FLUSH + TOP_FIVE_CARDS_TABLE[ss]
        elif N_BITS_TABLE[sc] >= 5:
            if STRAIGHT_TABLE[sc] != 0:
                return HANDTYPE_VALUE_STRAIGHTFLUSH + <unsigned int>(STRAIGHT_TABLE[sc] << TOP_CARD_SHIFT)
            else:
                retval = HANDTYPE_VALUE_FLUSH + TOP_FIVE_CARDS_TABLE[sc]
        elif N_BITS_TABLE[sd] >= 5:
            if STRAIGHT_TABLE[sd] != 0:
                return HANDTYPE_VALUE_STRAIGHTFLUSH + <unsigned int>(STRAIGHT_TABLE[sd] << TOP_CARD_SHIFT)
            else:
                retval = HANDTYPE_VALUE_FLUSH + TOP_FIVE_CARDS_TABLE[sd]
        elif N_BITS_TABLE[sh] >= 5:
            if STRAIGHT_TABLE[sh] != 0:
                return HANDTYPE_VALUE_STRAIGHTFLUSH + <unsigned int>(STRAIGHT_TABLE[sh] << TOP_CARD_SHIFT)
            else:
                retval = HANDTYPE_VALUE_FLUSH + TOP_FIVE_CARDS_TABLE[sh]
        else:
            st = STRAIGHT_TABLE[ranks]
            if st != 0:
                retval = HANDTYPE_VALUE_STRAIGHT + (st << TOP_CARD_SHIFT)

        if retval != 0 and n_dups < 3:
            return retval

    if n_dups == 0:
        return HANDTYPE_VALUE_HIGHCARD + TOP_FIVE_CARDS_TABLE[ranks]
    elif n_dups == 1:
        two_mask = ranks ^ (sc ^ sd ^ sh ^ ss)
        retval = <unsigned int>(HANDTYPE_VALUE_PAIR + (TOP_CARD_TABLE[two_mask] << TOP_CARD_SHIFT))
        t = ranks ^ two_mask
        kickers = (TOP_FIVE_CARDS_TABLE[t] >> CARD_WIDTH) & ~FIFTH_CARD_MASK
        retval += kickers
        retval |= (TOP_CARD_TABLE[t] << THIRD_CARD_SHIFT)  # Add kicker
        return retval
    elif n_dups == 2:
        two_mask = ranks ^ (sc ^ sd ^ sh ^ ss)
        if two_mask != 0:
            t = ranks ^ two_mask
            retval = <unsigned int>(HANDTYPE_VALUE_TWOPAIR
                + (TOP_FIVE_CARDS_TABLE[two_mask]
                & (TOP_CARD_MASK | SECOND_CARD_MASK))
                + (TOP_CARD_TABLE[t] << THIRD_CARD_SHIFT))
            return retval
        else:
            three_mask = ((sc & sd) | (sh & ss)) & ((sc & sh) | (sd & ss))
            retval = <unsigned int>(HANDTYPE_VALUE_TRIPS + (TOP_CARD_TABLE[three_mask] << TOP_CARD_SHIFT))
            t = ranks ^ three_mask
            second = TOP_CARD_TABLE[t]
            retval += (second << SECOND_CARD_SHIFT)
            t ^= (1U << <int>second)
            retval += <unsigned int>(TOP_CARD_TABLE[t] << THIRD_CARD_SHIFT)
            return retval
    else:
        four_mask = sh & sd & sc & ss
        if four_mask != 0:
            tc = TOP_CARD_TABLE[four_mask]
            retval = <unsigned int>(HANDTYPE_VALUE_FOUR_OF_A_KIND
                + (tc << TOP_CARD_SHIFT)
                + ((TOP_CARD_TABLE[ranks ^ (1U << <int>tc)]) << SECOND_CARD_SHIFT))
            return retval
        two_mask = ranks ^ (sc ^ sd ^ sh ^ ss)
        if N_BITS_TABLE[two_mask] != n_dups:
            three_mask = ((sc & sd) | (sh & ss)) & ((sc & sh) | (sd & ss))
            retval = HANDTYPE_VALUE_FULLHOUSE
            tc = TOP_CARD_TABLE[three_mask]
            retval += (tc << TOP_CARD_SHIFT)
            t = (two_mask | three_mask) ^ (1U << <int>tc)
            retval += <unsigned int>(TOP_CARD_TABLE[t] << SECOND_CARD_SHIFT)
            return retval
        if retval != 0:
            return retval
        else:
            retval = HANDTYPE_VALUE_TWOPAIR
            top = TOP_CARD_TABLE[two_mask]
            retval += (top << TOP_CARD_SHIFT)
            second = TOP_CARD_TABLE[two_mask ^ (1 << <int>top)]
            retval += (second << SECOND_CARD_SHIFT)
            retval += <unsigned int>((TOP_CARD_TABLE[ranks ^ (1U << <int>top) ^ (1 << <int>second)]) << THIRD_CARD_SHIFT)
            return retval


""" TODO: Leverage for abstractions
Heres the idea: 
    On the flop, you have 5 cards, run it through cy_evaluate to get a hand value.
     and then determin the hand type using this function. Next, draw a card from the deck
     and determin the hand type. Do this for all remaining cards; mapping hand type to count.



NOTE EXAMPLE  USAGE NOTE
print(handtype(card_str_to_int("AC") | card_str_to_int("AD") | card_str_to_int("4C") | card_str_to_int("5S") | card_str_to_int("JH") | card_str_to_int("JC") | card_str_to_int("JD"), 7))
NOTE NOTE NOTE NOTE NOTE
"""
cpdef handtype(unsigned long long hand_board, unsigned int num_cards):
    # Call cystom evaluate for handtyping.
    cdef unsigned int value = cy_evaluate_handtype(hand_board, num_cards)
    cdef unsigned int ht = (value >> HANDTYPE_SHIFT)
    top_card = VALUES[(value & TOP_CARD_MASK) >> TOP_CARD_SHIFT]
    second_card = VALUES[(value & SECOND_CARD_MASK) >> SECOND_CARD_SHIFT]
    third_card = VALUES[(value & THIRD_CARD_SHIFT) >> THIRD_CARD_SHIFT]
    
    if ht == HANDTYPE_VALUE_HIGHCARD >> HANDTYPE_SHIFT:
        return f"High Card, top card: {top_card}"
    elif ht == HANDTYPE_VALUE_PAIR >> HANDTYPE_SHIFT:
        return f"Pair of {top_card}s, kicker: {second_card}"
    elif ht == HANDTYPE_VALUE_TWOPAIR >> HANDTYPE_SHIFT:
        return f"Two Pair, {top_card}s and {second_card}s, kicker: {third_card}"
    elif ht == HANDTYPE_VALUE_TRIPS >> HANDTYPE_SHIFT:
        return f"Trips of {top_card}s, kicker: {second_card}"
    elif ht == HANDTYPE_VALUE_STRAIGHT >> HANDTYPE_SHIFT:
        return f"Straight to {top_card}"
    elif ht == HANDTYPE_VALUE_FLUSH >> HANDTYPE_SHIFT:
        return f"Flush to {top_card}"
    elif ht == HANDTYPE_VALUE_FULLHOUSE >> HANDTYPE_SHIFT:
        return f"Full House, {top_card}s full of {second_card}s"
    elif ht == HANDTYPE_VALUE_FOUR_OF_A_KIND >> HANDTYPE_SHIFT:
        return f"Four of a Kind, {top_card}s, kicker: {second_card}"
    else:
        return f"Straight Flush to {top_card}"

cpdef handtype_partial(unsigned long long hand_board, unsigned int num_cards):
    # Call cystom evaluate for handtyping.
    cdef unsigned int value = cy_evaluate_handtype(hand_board, num_cards)
    cdef unsigned int ht = (value >> HANDTYPE_SHIFT)
    top_card = VALUES[(value & TOP_CARD_MASK) >> TOP_CARD_SHIFT]
    second_card = VALUES[(value & SECOND_CARD_MASK) >> SECOND_CARD_SHIFT]
    # third_card = VALUES[(value & THIRD_CARD_SHIFT) >> THIRD_CARD_SHIFT]
    
    if ht == HANDTYPE_VALUE_HIGHCARD >> HANDTYPE_SHIFT:
        return f"High Card"
    elif ht == HANDTYPE_VALUE_PAIR >> HANDTYPE_SHIFT:
        return f"Pair"
    elif ht == HANDTYPE_VALUE_TWOPAIR >> HANDTYPE_SHIFT:
        return f"Two Pair"
    elif ht == HANDTYPE_VALUE_TRIPS >> HANDTYPE_SHIFT:
        return f"Trips"
    elif ht == HANDTYPE_VALUE_STRAIGHT >> HANDTYPE_SHIFT:
        return f"Straight"
    elif ht == HANDTYPE_VALUE_FLUSH >> HANDTYPE_SHIFT:
        return f"Flush"
    elif ht == HANDTYPE_VALUE_FULLHOUSE >> HANDTYPE_SHIFT:
        return f"Full House"
    elif ht == HANDTYPE_VALUE_FOUR_OF_A_KIND >> HANDTYPE_SHIFT:
        return f"Four of a Kind"
    else:
        return f"Straight Flush"

## TODO: Translate to cython environment using memeoryviews and Tie/Win/Loss tally for exact winrates.
##    The end goal for abstraction is to be able to:
##      1. (Baseline) Take a gamestate and abstract it.
##      2. Determine all remaining, possible, and abstracted gamestates can come from the current gamestate
##      3. Use all possible emission states (or ground truths if calculating the river) to 'classify' the current hand.
##      Conclusion. The current hand is abstracted based on all possible emission states from that hand.
##                  These emission states are abstractions of the hand/(new)board based on the handtype.
##                          **On the river, abstracted states emit Expected Values.
##                  Taking some (lots of flexibility here) aggregate of the emissions for the current gamestate
##                  will result in a non-unique, but extremely discriptive, view of the current gamestate.
##                          ** By adjusting _utils.handtype and the aggregator for abstracted emissions states one can tune
##                              a balance of abstraction and uniqueness.
""" NOTE: PSEUDOCODE FROM CHATGPT:
### TODO: implmenet this skeleton in my environment.

# Step 1: Define hand types and their values
HAND_TYPES = ["High Card", "Pair", "Two Pair", "Trips", "Straight", "Flush", "Full House", "Four of a Kind", "Straight Flush"]

def evaluate_hand(hand):
    Evaluates the hand and returns its type and relevant details.
    # Use cy_evaluate or similar function to get hand value
    value = cy_evaluate_handtype(hand, len(hand))
    hand_type = value >> HANDTYPE_SHIFT
    details = extract_hand_details(value)
    return HAND_TYPES[hand_type], details

# Step 2: Calculate River Abstractions
def calculate_river_abstractions(deck, board):
    river_abstractions = {}
    for river_card in deck:
        hand = board + [river_card]
        hand_type, details = evaluate_hand(hand)
        if hand_type not in river_abstractions:
            river_abstractions[hand_type] = []
        river_abstractions[hand_type].append(details)
    return river_abstractions

# Step 3: Calculate Turn Abstractions
def calculate_turn_abstractions(deck, board):
    turn_abstractions = {}
    for turn_card in deck:
        remaining_deck = [card for card in deck if card != turn_card]
        turn_board = board + [turn_card]
        river_abstractions = calculate_river_abstractions(remaining_deck, turn_board)
        hand_type, details = evaluate_hand(turn_board)
        if hand_type not in turn_abstractions:
            turn_abstractions[hand_type] = []
        turn_abstractions[hand_type].append({
            "turn_details": details,
            "river_abstractions": river_abstractions
        })
    return turn_abstractions

# Step 4: Calculate Flop Abstractions
def calculate_flop_abstractions(deck, board):
    flop_abstractions = {}
    for flop_card in deck:
        remaining_deck = [card for card in deck if card != flop_card]
        flop_board = board + [flop_card]
        turn_abstractions = calculate_turn_abstractions(remaining_deck, flop_board)
        hand_type, details = evaluate_hand(flop_board)
        if hand_type not in flop_abstractions:
            flop_abstractions[hand_type] = []
        flop_abstractions[hand_type].append({
            "flop_details": details,
            "turn_abstractions": turn_abstractions
        })
    return flop_abstractions

# Step 5: Aggregate Results
def aggregate_results(deck, board):
    flop_abstractions = calculate_flop_abstractions(deck, board)
    # Aggregate and analyze results as needed
    return flop_abstractions

# Example usage
deck = [card for card in range(52)]  # Simplified deck representation
board = [card1, card2, card3]  # Initial board cards (flop)
abstractions = aggregate_results(deck, board)
print(abstractions)
"""



'''
Enable calling cy-evaluate from the main process for testing purposes.
'''

cpdef cy_evaluate_cpp(cards, num_cards):
    cdef unsigned long long crds = cards
    cdef unsigned long long num_crds = num_cards
    return cy_evaluate(crds, num_crds)









def _6_max_opening():
    # Unopened ranges (Early to Late)
    UTG_OPEN = []
    HJ_OPEN = [('fold', 0)]  # Assuming UTG has folded
    CO_OPEN = [('fold', 0), ('fold', 0)]  # Assuming UTG and HJ have folded
    BTN_OPEN = [('fold', 0), ('fold', 0), ('fold', 0)]  # Assuming UTG, HJ, and CO have folded
    SB_OPEN = [('fold', 0), ('fold', 0), ('fold', 0), ('fold', 0)]  # Assuming UTG, HJ, CO, and BTN have folded

    # Defensive ranges (Early vs. Early to Late vs. Late)
    HJ_DEF = [('raise', 1.5)]  # Assuming UTG has raised
    CO_UTG_DEF = [('raise', 1.5), ('fold', 0)]  # Assuming UTG has raised, HJ has folded
    CO_HJ_DEF = [('fold', 0), ('raise', 1.5)]  # Assuming UTG has folded, HJ has raised
    BTN_UTG_DEF = [('raise', 1.5), ('fold', 0), ('fold', 0)]  # Assuming UTG has raised, HJ and CO have folded
    BTN_HJ_DEF = [('fold', 0), ('raise', 1.5), ('fold', 0)]  # Assuming UTG has folded, HJ has raised, CO has folded
    BTN_CO_DEF = [('fold', 0), ('fold', 0), ('raise', 1.5)]  # Assuming UTG and HJ have folded, CO has raised
    SB_UTG_DEF = [('raise', 1.5), ('fold', 0), ('fold', 0), ('fold', 0)]  # Assuming UTG has raised, HJ, CO, and BTN have folded
    SB_HJ_DEF = [('fold', 0), ('raise', 1.5), ('fold', 0), ('fold', 0)]  # Assuming UTG has folded, HJ has raised, CO and BTN have folded
    SB_CO_DEF = [('fold', 0), ('fold', 0), ('raise', 1.5), ('fold', 0)]  # Assuming UTG and HJ have folded, CO has raised, BTN has folded
    SB_BTN_DEF = [('fold', 0), ('fold', 0), ('fold', 0), ('raise', 1.5)]  # Assuming UTG, HJ, and CO have folded, BTN has raised

    BB_DEF = [('raise', 1.5), ('fold', 0), ('fold', 0), ('fold', 0), ('fold', 0)]  # Assuming UTG has raised, HJ, CO, BTN, and SB have folded
    BB_HJ_DEF = [('fold', 0), ('raise', 1.5), ('fold', 0), ('fold', 0), ('fold', 0)]  # Assuming UTG has folded, HJ has raised, CO, BTN, and SB have folded
    BB_CO_DEF = [('fold', 0), ('fold', 0), ('raise', 1.5), ('fold', 0), ('fold', 0)]  # Assuming UTG and HJ have folded, CO has raised, BTN and SB have folded
    BB_BTN_DEF = [('fold', 0), ('fold', 0), ('fold', 0), ('raise', 1.5), ('fold', 0)]  # Assuming UTG, HJ, and CO have folded, BTN has raised, SB has folded
    BB_SB_DEF = [('fold', 0), ('fold', 0), ('fold', 0), ('fold', 0), ('raise', 1.5)]  # Assuming UTG, HJ, CO, and BTN have folded, SB has raised

    # Three-bet ranges
    HJ_UTG_3BET = [('raise', 1.5)] + [('fold', 0)] * 5 + [('raise', 2.0)]  # HJ three-bet vs. UTG open
    CO_UTG_3BET = [('raise', 1.5), ('fold', 0)] + [('fold', 0)] * 4 + [('raise', 2.0)]  # CO three-bet vs. UTG open
    CO_HJ_3BET = [('fold', 0), ('raise', 1.5)] + [('fold', 0)] * 4 + [('raise', 2.0)]  # CO three-bet vs. HJ open
    BTN_UTG_3BET = [('raise', 1.5), ('fold', 0), ('fold', 0)] + [('fold', 0)] * 3 + [('raise', 2.0)]  # BTN three-bet vs. UTG open
    BTN_HJ_3BET = [('fold', 0), ('raise', 1.5), ('fold', 0)] + [('fold', 0)] * 3 + [('raise', 2.0)]  # BTN three-bet vs. HJ open
    BTN_CO_3BET = [('fold', 0), ('fold', 0), ('raise', 1.5)] + [('fold', 0)] * 3 + [('raise', 2.0)]  # BTN three-bet vs. CO open
    SB_UTG_3BET = [('raise', 1.5), ('fold', 0), ('fold', 0), ('fold', 0)] + [('fold', 0)] * 2 + [('raise', 2.0)]  # SB three-bet vs. UTG open
    SB_HJ_3BET = [('fold', 0), ('raise', 1.5), ('fold', 0), ('fold', 0)] + [('fold', 0)] * 2 + [('raise', 2.0)]  # SB three-bet vs. HJ open
    SB_CO_3BET = [('fold', 0), ('fold', 0), ('raise', 1.5), ('fold', 0)] + [('fold', 0)] * 2 + [('raise', 2.0)]  # SB three-bet vs. CO open
    SB_BTN_3BET = [('fold', 0), ('fold', 0), ('fold', 0), ('raise', 1.5)] + [('fold', 0)] * 2 + [('raise', 2.0)]  # SB three-bet vs. BTN open
    BB_UTG_3BET = [('raise', 1.5), ('fold', 0), ('fold', 0), ('fold', 0), ('fold', 0)] + [('fold', 0)] + [('raise', 2.0)]  # BB three-bet vs. UTG open
    BB_HJ_3BET = [('fold', 0), ('raise', 1.5), ('fold', 0), ('fold', 0), ('fold', 0)] + [('fold', 0)] + [('raise', 2.0)]  # BB three-bet vs. HJ open
    BB_CO_3BET = [('fold', 0), ('fold', 0), ('raise', 1.5), ('fold', 0), ('fold', 0)] + [('fold', 0)] + [('raise', 2.0)]  # BB three-bet vs. CO open
    BB_BTN_3BET = [('fold', 0), ('fold', 0), ('fold', 0), ('raise', 1.5), ('fold', 0)] + [('fold', 0)] + [('raise', 2.0)]  # BB three-bet vs. BTN open
    BB_SB_3BET = [('fold', 0), ('fold', 0), ('fold', 0), ('fold', 0), ('raise', 1.5)] + [('fold', 0)] + [('raise', 2.0)]  # BB three-bet vs. SB open

    positions_to_solve = [
        UTG_OPEN,
        HJ_OPEN, 
        CO_OPEN,  
        BTN_OPEN, 
        SB_OPEN, 
        HJ_DEF,  
        CO_UTG_DEF,
        CO_HJ_DEF, 
        BTN_UTG_DEF, 
        BTN_HJ_DEF, 
        BTN_CO_DEF, 
        SB_UTG_DEF, 
        SB_HJ_DEF, 
        SB_CO_DEF,
        SB_BTN_DEF,
        BB_DEF,
        BB_HJ_DEF,
        BB_CO_DEF,
        BB_BTN_DEF,
        BB_SB_DEF,
        HJ_UTG_3BET,
        CO_UTG_3BET,
        CO_HJ_3BET,
        BTN_UTG_3BET,
        BTN_HJ_3BET,
        BTN_CO_3BET,
        SB_UTG_3BET,
        SB_HJ_3BET,
        SB_CO_3BET,
        SB_BTN_3BET,
        BB_UTG_3BET,
        BB_HJ_3BET,
        BB_CO_3BET,
        BB_BTN_3BET,
        BB_SB_3BET
    ]

    position_names = [
        "UTG_OPEN", 
        "HJ_OPEN", 
        "CO_OPEN",  
        "BTN_OPEN", 
        "SB_OPEN", 
        "HJ_DEF",  
        "CO_UTG_DEF",
        "CO_HJ_DEF", 
        "BTN_UTG_DEF", 
        "BTN_HJ_DEF",
        "BTN_CO_DEF", 
        "SB_UTG_DEF", 
        "SB_HJ_DEF", 
        "SB_CO_DEF",
        "SB_BTN_DEF",
        "BB_DEF",
        "BB_HJ_DEF",
        "BB_CO_DEF",
        "BB_BTN_DEF",
        "BB_SB_DEF",
        "HJ_UTG_3BET",
        "CO_UTG_3BET",
        "CO_HJ_3BET",
        "BTN_UTG_3BET",
        "BTN_HJ_3BET",
        "BTN_CO_3BET",
        "SB_UTG_3BET",
        "SB_HJ_3BET",
        "SB_CO_3BET",
        "SB_BTN_3BET",
        "BB_UTG_3BET",
        "BB_HJ_3BET",
        "BB_CO_3BET",
        "BB_BTN_3BET",
        "BB_SB_3BET"
    ]
    
    positions_dict = {str(pos): name for pos, name in zip(positions_to_solve, position_names)}

    return positions_to_solve, positions_dict
