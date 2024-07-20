import os
import random
import numpy
from tqdm import tqdm


cdef public list SUITS = ['C', 'D', 'H', 'S']
cdef public list VALUES = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']

cdef public dict SUITS_INDEX = {'C': 0, 'D': 1, 'H': 2, 'S': 3}
cdef public dict VALUES_INDEX = {'2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6, '9': 7, 'T': 8, 'J': 9, 'Q': 10, 'K': 11, 'A': 12}



cpdef dynamic_merge_dicts(local_manager_table, global_accumulator):
    
    for player_hash_local in local_manager_table.to_merge:
        inner_dict = local_manager_table.get_hashed(player_hash_local)
        if player_hash_local in global_accumulator:
            existing_inner_dict = global_accumulator[player_hash_local]

            for inner_key, inner_value in inner_dict.items():
                if inner_key in existing_inner_dict:
                    existing_inner_dict[inner_key] = (existing_inner_dict[inner_key] + inner_value)
                else:
                    existing_inner_dict[inner_key] = inner_value

            global_accumulator[player_hash_local] = existing_inner_dict
        else:
            global_accumulator[player_hash_local] = inner_dict
        
        
        # TODO: Reassigning to the inner values is a bold move.
        ##      Investigate accumulating results.
        ##   Think: If we're batch processing hands and we accumulate by reasignment.. 
        ## Then, for a given batch, we'll take the results from only the last abstracted state.


##########


cpdef list build_fast_forward_actions(list action_space):
    cdef list fast_forward_actions = action_space[0][2:] + action_space[1] + action_space[2] + action_space[3]
    
    # Sort of confusing function.. hence why it's a utility.
    return [pos_w_action[1] for pos_w_action in fast_forward_actions]


cpdef object select_random_action(average_strategy):
    actions = list(average_strategy.keys())
    probabilities = list(average_strategy.values())
    selected_action = random.choices(actions, probabilities)[0]
    return selected_action


##########


cdef list deck = [card_to_int(suit, value) for suit in SUITS for value in VALUES]


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


cdef tuple ulong_to_card_tuple(unsigned long long hand):
    cards = [card for card in deck if card & hand]
    return tuple(cards)

cdef tuple card_tuple_to_str_tuple(tuple cards):
    return tuple([int_to_card(card) for card in cards])

cdef str format_hand(unsigned long long hand):
    return " ".join(card_tuple_to_str_tuple(ulong_to_card_tuple(hand)))


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
    last_move = next((item for sublist in reversed(game_state.action_space) for item in reversed(sublist) if item is not None and (item[0] not in folded or item[1][0] == 'fold')), None)
    
    header = (
        f"______________________________________________________________________________\n"
        f"({current_player.position}): {format_hand(current_player.hand)} --- {'folded' if current_player.folded else 'active'}\n"
        f"Board: {format_hand(game_state.board)}\n"
        f"Pot: {game_state.pot}\n"
        f"Chips: {current_player.chips}\n"
        f"Last move: {last_move}\n"
        f"______________________________________________________________________________\n"
    )
    return header

# Helper function to display the betting rounds
def display_betting_rounds(game_state):
    betting_rounds = (
        f'          {" " * game_state.cur_round_index * 20}|\n'
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

    for round_idx, round_actions in enumerate(game_state.action_space):
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
    
    print(f"\nPOS      CARDS    POT CONTRIBS    STATUS     STACK     PRIOR GAINS")
    for i, player in enumerate(game_state.players):
        print(display_player_info(player, game_state, current_player, player_index))
    print(f"______________________________________________________________________________")
    
    print(display_betting_rounds(game_state))
    
    actions_dict, rounds = generate_actions_dict(game_state, folded)
    print(display_actions(actions_dict, rounds))

    print(display_header(game_state, current_player))




def fold_list(count):
    return [('fold', 0)] * count

def call_list(count):
    return [('call', 0)] * count

def _6_max_opening():

    ranges = {
        ### NOTE: SB
        "SB_OPEN": fold_list(4),
        "BB_SB_DEF": fold_list(4) + [('raise', 1.5)],
        "SB_BB_3B_DEF": fold_list(4) + [('raise', 1.5)] + [('raise', 2.0)] + call_list(4),

        ### NOTE: BTN
        "BTN_OPEN": fold_list(3),
        "SB_BTN_DEF": fold_list(3) + [('raise', 1.5)],
        "BB_BTN_DEF": fold_list(3) + [('raise', 1.5)] + fold_list(1),
        "BTN_SB_3B_DEF": fold_list(3) + [('raise', 1.5)] + [('raise', 2.0)] + fold_list(1) + call_list(3),
        "BTN_BB_3B_DEF": fold_list(3) + [('raise', 1.5)] + fold_list(1) + [('raise', 2.0)] + call_list(3),

        ### NOTE: CO
        "CO_OPEN": fold_list(2),
        "BTN_CO_DEF": fold_list(2) + [('raise', 1.5)],
        "SB_CO_DEF": fold_list(2) + [('raise', 1.5)] + fold_list(1),
        "BB_CO_DEF": fold_list(2) + [('raise', 1.5)] + fold_list(2),
        "CO_BTN_3B_DEF": fold_list(2) + [('raise', 1.5)] + [('raise', 2.0)] + fold_list(2) + call_list(2),
        "CO_SB_3B_DEF": fold_list(2) + [('raise', 1.5)] + fold_list(1) + [('raise', 2.0)] + fold_list(1) + call_list(2),
        "CO_BB_3B_DEF": fold_list(2) + [('raise', 1.5)] + fold_list(2) + [('raise', 2.0)] + call_list(2),

        ### NOTE: MP
        "MP_OPEN": fold_list(1),
        "CO_MP_DEF": fold_list(1) + [('raise', 1.5)],
        "BTN_MP_DEF": fold_list(1) + [('raise', 1.5)] + fold_list(1),
        "SB_MP_DEF": fold_list(1) + [('raise', 1.5)] + fold_list(2),
        "BB_MP_DEF": fold_list(1) + [('raise', 1.5)] + fold_list(3),
        "MP_CO_3B_DEF": fold_list(1) + [('raise', 1.5), ('raise', 2.0)] + fold_list(3) + call_list(1),
        "MP_BTN_3B_DEF": fold_list(1) + [('raise', 1.5)] + fold_list(1) + [('raise', 2.0)] + fold_list(2) + call_list(1),
        "MP_SB_3B_DEF": fold_list(1) + [('raise', 1.5)] + fold_list(2) + [('raise', 2.0)] + fold_list(1) + call_list(1),
        "MP_BB_3B_DEF": fold_list(1) + [('raise', 1.5)] + fold_list(3) + [('raise', 2.0)] + call_list(1),

        ### NOTE: UTG
        "UTG_OPEN": [],
        "MP_UTG_DEF": [('raise', 1.5)],
        "CO_UTG_DEF": [('raise', 1.5)] + fold_list(1),
        "BTN_UTG_DEF": [('raise', 1.5)] + fold_list(2),
        "SB_UTG_DEF": [('raise', 1.5)] + fold_list(3),
        "BB_UTG_DEF": [('raise', 1.5)] + fold_list(4),
        "UTG_MP_3B_DEF": [('raise', 1.5), ('raise', 2.0)] + fold_list(4),
        "UTG_CO_3B_DEF": [('raise', 1.5)] + fold_list(1) + [('raise', 2.0)] + fold_list(3),
        "UTG_BTN_3B_DEF": [('raise', 1.5)] + fold_list(2) + [('raise', 2.0)] + fold_list(2),
        "UTG_SB_3B_DEF": [('raise', 1.5)] + fold_list(3) + [('raise', 2.0)] + fold_list(1),
        "UTG_BB_3B_DEF": [('raise', 1.5)] + fold_list(4) + [('raise', 2.0)],
    }

    # Generate the positions to solve and their names
    positions_to_solve = list(ranges.values())
    position_names = list(ranges.keys())

    # Create dictionary mapping each position to its name and range
    positions_dict = {str(pos): name for name, pos in ranges.items()}

    return positions_to_solve, positions_dict

def _2_max_opening():

    ranges = {
        ### NOTE: SB
        "SB_OPEN": [],
        "BB_SB_DEF": [('raise', 1.5)],
        "SB_BB_3B_DEF": [('raise', 1.5)] + [('raise', 2.0)],
        "BB_SB_4B_DEF": [('raise', 1.5)] + [('raise', 2.0)] + [('raise', 2.0)],
    }

    # Generate the positions to solve and their names
    positions_to_solve = list(ranges.values())
    position_names = list(ranges.keys())

    # Create dictionary mapping each position to its name and range
    positions_dict = {str(pos): name for name, pos in ranges.items()}

    return positions_to_solve, positions_dict

def _6_max_simple_postflop():

    # Unopened ranges (Early to Late)
    ranges = {"SB_BB_3B_DEF_POSTFLOP": fold_list(4) + [('raise', 1.5)] + [('raise', 2.0)] + call_list(4) + call_list(1)} # add additional call for SB action. result in post flop

    # Generate the positions to solve and their names
    positions_to_solve = list(ranges.values())
    position_names = list(ranges.keys())

    # Create dictionary mapping each position to its name and range
    positions_dict = {str(pos): name for name, pos in ranges.items()}

    return positions_to_solve, positions_dict