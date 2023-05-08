#!python
#cython: language_level=3

cimport numpy
cimport cython


from libc.stdio cimport FILE, fopen, fwrite, fscanf, fclose, fprintf
# cython: profile=True
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils import check_array

from libc.stdlib cimport rand, srand

cdef public list SUITS = ['C', 'D', 'H', 'S']
cdef public list VALUES = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']


cdef class GameState:
    def __init__(self, list players, int initial_chips, int small_blind, int big_blind):
        self.players = players
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.dealer_position = 0
        self.player_index = (self.dealer_position + 3) % len(self.players)
        self.pot = 0
        self.current_bet = 0
        self.board = 0
        self.deck = create_deck()
        self.fisher_yates_shuffle()

    cpdef reset(self):
        self.deck = create_deck()
        self.fisher_yates_shuffle()

        self.pot = 0
        self.current_bet = 0
        self.board = 0
        for player in self.players:
            player.reset()

    cpdef clone(self):
        cdef GameState new_state = GameState(self.players[:], self.small_blind, self.big_blind)
        new_state.dealer_position = self.dealer_position
        new_state.pot = self.pot
        new_state.current_bet = self.current_bet
        new_state.board = self.board
        new_state.deck = self.deck[:]
        return new_state

    cpdef handle_blinds(self):
        cdef int small_blind_pos = (self.dealer_position + 1) % len(self.players)
        cdef int big_blind_pos = (self.dealer_position + 2) % len(self.players)
        
        self.players[small_blind_pos].player_action(self, small_blind_pos, "raise", min(self.small_blind, self.players[small_blind_pos].chips))
        self.players[big_blind_pos].player_action(self, big_blind_pos, "raise", min(self.small_blind, self.players[big_blind_pos].chips))

    
    cpdef preflop(self):
        starting_player = (self.dealer_position + 3) % len(self.players)
        self.player_index = starting_player
        
        order = list(range(starting_player, len(self.players))) + list(range(0, starting_player))
        last_raiser, num_actions, index = -1, 0, 0
        while ((num_actions < len(self.players)) or last_raiser != self.player_index) and self.active_players() > 1:
            self.player_index = order[index % len(self.players)]
            if self.players[self.player_index].get_action(self, self.player_index):
                last_raiser = self.player_index
            num_actions += 1
            index += 1
            self.player_index = order[index % len(self.players)]
            
            if(num_actions >= len(self.players) and (last_raiser == -1 or last_raiser == self.player_index)):
                break

    cpdef postflop(self, str round_name):
        # Between rounds, we want reset the current_bet and pot contribution numbers.
        self.current_bet = 0
        for i in range(len(self.players)):
            self.players[i].contributed_to_pot = 0
        
        if round_name == "flop":
            for _ in range(3):
                self.draw_card()
        else:
            self.draw_card()

        starting_player = (self.dealer_position + 1) % len(self.players)
        self.player_index = starting_player
        
        order = list(range(starting_player, len(self.players))) + list(range(0, starting_player))
        
        last_raiser, num_actions, index = -1, 0, 0
        while ((num_actions < len(self.players)) or last_raiser != self.player_index) and self.active_players() > 1:
            self.player_index = order[index % len(self.players)]
            if self.players[self.player_index].get_action(self, self.player_index):
                last_raiser = self.player_index
            num_actions += 1
            index += 1
            self.player_index = order[index % len(self.players)]
            
            if(num_actions >= len(self.players) and (last_raiser == -1 or last_raiser == self.player_index)):
                break


    cpdef showdown(self):
        cdef unsigned long long player_hand
        cdef int best_score, player_score, winner_index

        cdef int remaining_players = sum([not player.folded for player in self.players])
        if remaining_players == 1:
            for i, player in enumerate(self.players):
                if not player.folded:
                    winner_index = i
                    break
        else:

            best_score = -1
            winner_index = -1

            for i, player in enumerate(self.players):
                player_hand = player.hand | self.board
                player_score = cy_evaluate(player_hand | self.board, 7)

                if player_score > best_score:
                    best_score = player_score
                    winner_index = i

            # Distribute the pot to the winner
        self.players[winner_index].chips += self.pot
        self.pot = 0

    

    cpdef bint is_terminal(self):
        cdef int active_players = 0
        cdef bint all_acted
        #cdef Player player
        #cdef int num_active_players = sum([1 for player in self.players if (not player.folded and player.chips > 0)])
        for player in self.players:
            if not player.folded and player.chips > 0:
                active_players += 1
            if active_players > 1:
                break
        
        # Terminal case: all but one player have folded
        if active_players == 1:
            return True

        # Terminal case: all 5 table cards have been dealt and players have acted
        if self.board_has_five_cards():
            all_acted = True
            for player in self.players:
                if not player.folded and player.contributed_to_pot < self.current_bet:
                    all_acted = False
                    break

            if all_acted:
                return True

        return False
        
    cpdef deal_private_cards(self):
        for player in self.players:
            if player.chips > 0:
                player.add_card(self.deck.pop())
                player.add_card(self.deck.pop())

    cpdef deal_public_cards(self):
        self.board |= self.deck.pop()  # Burn card before the flop
        for _ in range(3):  # Flop
            self.draw_card()
        self.draw_card()  # Turn
        self.draw_card()  # Burn card before the river
        self.draw_card()  # River
    
    cdef void fisher_yates_shuffle(self):
        cdef int i, j
        cdef unsigned long long temp
        srand(1)
        for i in range(len(self.deck) - 1, 0, -1):
            j = rand() % (i + 1)
            temp = self.deck[i]
            self.deck[i] = self.deck[j]
            self.deck[j] = temp

    cpdef int active_players(self):
        cdef int alive = 0
        for i in range(len(self.players)):
            if not self.players[i].folded:
                alive += 1
        return alive

    cpdef draw_card(self):
        self.board |= self.deck.pop()
        
    cdef bint board_has_five_cards(self):
        return bin(self.board).count('1') == 5


cpdef unsigned long long card_to_int(str suit, str value):
    cdef unsigned long long one = 1
    cdef int suit_index = SUITS.index(suit)
    cdef int value_index = VALUES.index(value)
    cdef int bit_position = suit_index * 13 + value_index
    return one << bit_position


cpdef public list create_deck():
        cdef list deck = [card_to_int(suit, value) for suit in SUITS for value in VALUES]
        return deck


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

cpdef str format_hand(unsigned long long hand):
    cdef list cards = [int_to_card(card) for card in create_deck() if card & hand]
    return " ".join(cards)

cpdef display_game_state(GameState game_state, int player_index):
    print(f"Player {player_index + 1}: {format_hand(game_state.players[player_index].hand)}")
    print(f"Board: {format_hand(game_state.board)}")
    print(f"Pot: {game_state.pot}")
    print(f"Chips: {game_state.players[player_index].chips}")


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