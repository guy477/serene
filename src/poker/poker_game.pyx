#!python
#cython: language_level=3


from libc.stdlib cimport rand, srand

cimport numpy
cimport cython


from libc.stdio cimport FILE, fopen, fwrite, fscanf, fclose, fprintf
# cython: profile=True
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils import check_array

ctypedef numpy.uint8_t uint8
ctypedef numpy.uint16_t uint16
ctypedef numpy.int16_t int16
ctypedef numpy.int64_t int64
ctypedef numpy.npy_bool boolean


#################################################################################################
# The below code is taken from eval7 - https://pypi.org/project/eval7/
#################################################################################################


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

#@cython.profile(True)
@cython.boundscheck(False) 
@cython.wraparound(False)
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




'''
Enable calling cy-evaluate from the main process for testing purposes.
'''
cpdef cy_evaluate_cpp(cards, num_cards):
    cdef unsigned long long crds = cards
    cdef unsigned long long num_crds = num_cards
    return cy_evaluate(crds, num_crds)


################################################################################################################
############################ POKER GAME ########################################################################
################################################################################################################


cdef list SUITS = ['C', 'D', 'H', 'S']
cdef list VALUES = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']

cdef class PokerGame:

    def __init__(self, int num_players, int initial_chips, int num_ai_players, int small_blind, int big_blind):
        self.players = [Player(initial_chips) for _ in range(num_players - num_ai_players)] + [AIPlayer(initial_chips) for _ in range(num_ai_players)]
        self.game_state = GameState(len(self.players), initial_chips, small_blind, big_blind)

    cpdef play_game(self, int num_hands=1):

        for _ in range(num_hands):
            self.deck = create_deck()
            fisher_yates_shuffle(self.deck)



            # Betting rounds
            preflop(self.game_state, self.deck)
            postflop(self.game_state, self.deck, "flop")
            postflop(self.game_state, self.deck, "turn")
            postflop(self.game_state, self.deck, "river")

            # Determine the winner and distribute the pot
            showdown(self.game_state)
            self.game_state.reset()

            # Update the dealer position
            self.game_state.dealer_position = (self.game_state.dealer_position + 1) % len(self.players)



cdef class GameState:

    def __init__(self, int num_players, int initial_chips, int small_blind, int big_blind):
        self.players = [Player(initial_chips) for _ in range(num_players)]
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.dealer_position = 0
        self.pot = 0
        self.current_bet = 0
        self.board = 0

    cpdef reset(self):
        self.pot = 0
        self.current_bet = 0
        self.board = 0
        for player in self.players:
            player.reset()


################################################################################

cpdef unsigned long long card_to_int(str suit, str value):
    cdef unsigned long long one = 1
    cdef int suit_index = SUITS.index(suit)
    cdef int value_index = VALUES.index(value)
    cdef int bit_position = suit_index * 13 + value_index
    return one << bit_position


cpdef list create_deck():
    cdef list deck = [card_to_int(suit, value) for suit in SUITS for value in VALUES]
    return deck


cdef void fisher_yates_shuffle(list deck):
    cdef int i, j
    cdef unsigned long long temp
    srand(1)
    for i in range(len(deck) - 1, 0, -1):
        j = rand() % (i + 1)
        temp = deck[i]
        deck[i] = deck[j]
        deck[j] = temp

cpdef unsigned long long draw_card(list deck):
    return deck.pop()

cpdef deal_cards(list deck, GameState game_state):
    cdef Player player
    for _ in range(2):
        for player in game_state.players:
            if player.chips > 0:
                player.add_card(draw_card(deck))


cpdef str int_to_card(unsigned long long card):
    cdef int bit_position = -1
    while card > 1:
        card >>= 1
        bit_position += 1
    cdef int suit_index = bit_position // 13
    cdef int value_index = bit_position % 13

    return f'{VALUES[value_index]}{SUITS[suit_index]}'

cpdef unsigned long long card_str_to_int(str card_str):
    return card_to_int(card_str[1], card_str[0])


cpdef showdown(GameState game_state):
    cdef Player player
    cdef unsigned long long player_hand
    cdef int best_score, player_score, winner_index

    cdef int remaining_players = sum([not player.folded for player in game_state.players])
    if remaining_players == 1:
        for i, player in enumerate(game_state.players):
            if not player.folded:
                winner_index = i
                break
    else:

        best_score = -1
        winner_index = -1

        for i, player in enumerate(game_state.players):
            player_hand = player.hand | game_state.board
            player_score = cy_evaluate(player_hand | game_state.board, 7)

            if player_score > best_score:
                best_score = player_score
                winner_index = i

        # Distribute the pot to the winner
    game_state.players[winner_index].chips += game_state.pot
    game_state.pot = 0


cdef handle_blinds(GameState game_state):
    cdef int small_blind_pos = (game_state.dealer_position + 1) % len(game_state.players)
    cdef int big_blind_pos = (game_state.dealer_position + 2) % len(game_state.players)

    player_action(game_state, small_blind_pos, "raise", min(game_state.small_blind, game_state.players[small_blind_pos].chips))
    player_action(game_state, big_blind_pos, "raise", min(game_state.small_blind, game_state.players[big_blind_pos].chips))

    # Update the current bet
    # TODO: Handle case where player's chips is less than the small-blind or big blind
    #       Player should still be able to play, but their chip count should not go negative.

cpdef preflop(GameState game_state, list deck):
    handle_blinds(game_state)  # Add this line
    deal_cards(deck, game_state)

    starting_player = (game_state.dealer_position + 3) % len(game_state.players)
    order = list(range(starting_player, len(game_state.players))) + list(range(0, starting_player))
    last_raiser, num_actions, index, player_index = -1, 0, 0, order[0]
    while ((num_actions < len(game_state.players)) or last_raiser != player_index) and active_players(game_state) > 1:
        player_index = order[index % len(game_state.players)]
        if process_user_input(game_state, player_index):
            last_raiser = player_index
        num_actions += 1
        index += 1
        player_index = order[index % len(game_state.players)]
        if(num_actions >= len(game_state.players) and (last_raiser == -1 or last_raiser == player_index)):
            break
        

cpdef postflop(GameState game_state, list deck, str round_name):
    # Between rounds, we want reset the current_bet and pot contribution numbers.
    game_state.current_bet = 0
    for i in range(len(game_state.players)):
        game_state.players[i].contributed_to_pot = 0
    
    if round_name == "flop":
        for _ in range(3):
            game_state.board |= draw_card(deck)
    else:
        game_state.board |= draw_card(deck)

    starting_player = (game_state.dealer_position + 1) % len(game_state.players)
    order = list(range(starting_player, len(game_state.players))) + list(range(0, starting_player))
    
    last_raiser, num_actions, index, player_index = -1, 0, 0, order[0]
    while ((num_actions < len(game_state.players)) or last_raiser != player_index) and active_players(game_state) > 1:
        player_index = order[index % len(game_state.players)]
        if process_user_input(game_state, player_index):
            last_raiser = player_index
        num_actions += 1
        index += 1
        player_index = order[index % len(game_state.players)]
        if(num_actions >= len(game_state.players) and (last_raiser == -1 or last_raiser == player_index)):
            break

cpdef int active_players(GameState game_state):
    cdef int alive = 0
    for i in range(len(game_state.players)):
        if not game_state.players[i].folded:
            alive += 1
    return alive

cpdef str format_hand(unsigned long long hand):
    cdef list cards = [int_to_card(card) for card in create_deck() if card & hand]
    return " ".join(cards)

cpdef display_game_state(GameState game_state, int player_index):
    print(f"Player {player_index + 1}: {format_hand(game_state.players[player_index].hand)}")
    print(f"Board: {format_hand(game_state.board)}")
    print(f"Pot: {game_state.pot}")
    print(f"Chips: {game_state.players[player_index].chips}")


################################################################################


cpdef player_action(GameState game_state, int player_index, str action, int bet_amount=0):
    cdef Player player = game_state.players[player_index]
    cdef int call_amount

    if player.folded or player.chips <= 0:
        return

    if action == "call":
        call_amount = game_state.current_bet - player.contributed_to_pot
        if call_amount > player.chips:
            call_amount = player.chips

        player.chips -= call_amount
        game_state.pot += call_amount
        player.contributed_to_pot += call_amount
        player.folded = False

        if player.chips <= 0:
            print(f"Player {player_index + 1} is out of chips.")

    elif action == "raise":
        if bet_amount < game_state.current_bet:
            if bet_amount != player.chips:
                raise ValueError("Raise amount must be at least equal to the current bet or an all-in.")
        
        call_amount = game_state.current_bet - player.contributed_to_pot
        if call_amount + bet_amount > player.chips:
            bet_amount = player.chips - call_amount

        player.chips -= (call_amount + bet_amount)
        game_state.pot += (call_amount + bet_amount)
        player.contributed_to_pot += (call_amount + bet_amount)
        game_state.current_bet += bet_amount
        player.folded = False

        if player.chips <= 0:
            print(f"Player {player_index + 1} is out of chips.")

    elif action == "fold":
        player.folded = True

    else:
        raise ValueError("Invalid action")


cpdef bint process_user_input(GameState game_state, int player_index):
    cdef str user_input
    cdef int bet_amount
    cdef bint valid = 0
    cdef bint raize = 0
    
    while valid == 0:
        if not game_state.players[player_index].folded:
            display_game_state(game_state, player_index)

            user_input = get_user_input("Enter action (call, raise, fold): ")
            
            if user_input == "call":
                player_action(game_state, player_index, "call")
                valid = 1
            elif user_input == "raise":
                bet_amount = int(get_user_input("Enter raise amount: "))
                player_action(game_state, player_index, "raise", bet_amount)
                raize = 1
                valid = 1
            elif user_input == "fold":
                player_action(game_state, player_index, "fold")
                valid = 1
            else:
                print("Invalid input. Please enter call, raise, or fold.")
        else:
            valid = 1
    return raize


cpdef str get_user_input(prompt):
    return input(prompt)    
