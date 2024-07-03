#!python
# cython: language_level=3
import numpy
cimport numpy
cimport cython

from libc.stdlib cimport RAND_MAX

import logging
import time


cdef public list SUITS = ['C', 'D', 'H', 'S']
cdef public list VALUES = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']

cdef public dict SUITS_INDEX = {'C': 0, 'D': 1, 'H': 2, 'S': 3}
cdef public dict VALUES_INDEX = {'2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6, '9': 7, 'T': 8, 'J': 9, 'Q': 10, 'K': 11, 'A': 12}

class PokerHandLogger:
    def __init__(self, log_file):
        self.logger = logging.getLogger('PokerHandLogger')
        self.logger.setLevel(logging.DEBUG)
        handler = logging.FileHandler(log_file)
        handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(handler)
        self.hand_count = 0

    def log_hand(self, hand_data):
        self.hand_count += 1
        self.logger.info(hand_data)

    def generate_hand_log(self, hand_id, game_type, limit, date_time, dealer_position, seats, blinds, hole_cards, actions, board, showdown, summary):
        hand_log = []
        hand_log.append(f"Hand #{hand_id} - {game_type} ({limit}) - {date_time} UTC")
        hand_log.append(f"Firestone 6-max Seat #{dealer_position} is the button")
        for seat, (player, chips) in enumerate(seats):
            hand_log.append(f"Seat {seat + 1}: {player} (${chips:.2f})")
        for blind, (player, amount) in blinds.items():
            hand_log.append(f"{player} posts the {blind} ${amount:.2f}")
        hand_log.append("*** HOLE CARDS ***")
        for player, cards in hole_cards.items():
            if cards:
                hand_log.append(f"Dealt to {player} [{cards[0]} {cards[1]}]")
            else:
                hand_log.append(f"Dealt to {player} [XX XX]")
        for round_name, round_actions in actions.items():
            if round_actions:
                hand_log.append(f"*** {round_name.upper()} ***")
                for action in round_actions:
                    hand_log.append(f"{action}")
        if board:
            hand_log.append(f"Board {' '.join(board)}")
        if showdown:
            hand_log.append("*** SHOW DOWN ***")
            for player, cards, description in showdown:
                if cards:
                    hand_log.append(f"{player} shows [{cards[0]} {cards[1]}] ({description})")
                else:
                    hand_log.append(f"{player} shows [XX XX] ({description})")
        hand_log.append("*** SUMMARY ***")
        for player, result in summary.items():
            hand_log.append(result)
        return '\n'.join(hand_log)

poker_logger = PokerHandLogger("poker_hand_history.log")

cdef class GameState:
    def __init__(self, list players, int small_blind, int big_blind, int num_simulations, bint silent=False, list suits=SUITS, list values=VALUES):
        self.players = players
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.num_simulations = num_simulations
        self.positions = self.generate_positions(len(players))
        self.suits = suits
        self.values = values
        self.silent = silent
        self.hand_id = 0

    cpdef reset(self):
        self.cur_round_index = 0
        self.dealer_position = 0
        self.player_index = (self.dealer_position + 3) % len(self.players)
        self.round_active_players = len(self.players)
        self.num_actions = 0
        self.last_raiser = -1
        self.current_bet = 0
        self.winner_index = -1
        self.pot = 0
        self.board = 0
        self.betting_history = [[], [], [], []]
        self.deck = self.create_deck(self.suits, self.values)
        self.fisher_yates_shuffle()
        for player in self.players:
            player.reset()

    cpdef log_current_hand(self, terminal=False):
        if self.silent:
            return
        
        seats = [(player.position, player.chips) for player in self.players]
        blinds = {
            "small blind": (self.players[(self.dealer_position + 1) % len(self.players)].position, self.small_blind),
            "big blind": (self.players[(self.dealer_position + 2) % len(self.players)].position, self.big_blind)
        }
        hole_cards = {player.position: [int_to_card(card) for card in hand_to_cards(player.hand)] for player in self.players if not player.folded}
        actions = {
            "pre-flop": self.betting_history[0],
            "flop": self.betting_history[1],
            "turn": self.betting_history[2],
            "river": self.betting_history[3]
        }
        board = [int_to_card(card) for card in hand_to_cards(self.board)]
        showdown = [(player.position, [int_to_card(card) for card in hand_to_cards(player.hand)], 'player.hand_description()') for player in self.players if not player.folded]
        summary = {player.position: 'player.result_description()' for player in self.players}
        hand_log = poker_logger.generate_hand_log(self.hand_id, "Holdem", "No Limit", time.strftime('%Y/%m/%d %H:%M:%S'), self.dealer_position + 1, seats, blinds, hole_cards, actions, board, showdown, summary)
        
        if terminal:
            poker_logger.log_hand(hand_log)
        
        self.hand_id += 1

    cpdef load_custom_betting_history(self, int round, object history):
        self.betting_history[round].append(history)

    cpdef handle_blinds(self):
        cdef int small_blind_pos = (self.dealer_position + 1) % len(self.players)
        cdef int big_blind_pos = (self.dealer_position + 2) % len(self.players)
        self.players[small_blind_pos].take_action(self, small_blind_pos, ("blinds", min(self.small_blind, self.players[small_blind_pos].chips)))
        self.players[big_blind_pos].take_action(self, big_blind_pos, ("blinds", min(self.big_blind, self.players[big_blind_pos].chips)))

    cpdef assign_positions(self):
        for i in range(len(self.players)):
            self.players[i].assign_position(self.positions[i], i)

    cpdef deal_private_cards(self, object hand=None):
        if hand:
            card1 = card_str_to_int(hand[0])
            card2 = card_str_to_int(hand[1])
            self.deck.remove(card1)
            self.deck.remove(card2)
            self.players[self.player_index].add_card(card1)
            self.players[self.player_index].add_card(card2)
            self.players[self.player_index].abstracted_hand = self.abstract_hand(card1, card2)
        for i, player in enumerate(self.players):
            if hand and i == self.player_index:
                continue
            if player.chips > 0:
                card1 = self.deck.pop()
                card2 = self.deck.pop()
                player.add_card(card1)
                player.add_card(card2)
                player.abstracted_hand = self.abstract_hand(card1, card2)
            else:
                player.folded = True

    cpdef str abstract_hand(self, unsigned long long card1, unsigned long long card2):
        cdef str card1_str = int_to_card(card1)
        cdef str card2_str = int_to_card(card2)

        # Temporary variables for the card values
        cdef str card1_val = card1_str[0]
        cdef str card2_val = card2_str[0]

        # Now use the temporary variables in your comparison
        cdef str high_card = card1_val if VALUES.index(card1_val) > VALUES.index(card2_val) else card2_val
        cdef str low_card = card1_val if VALUES.index(card1_val) < VALUES.index(card2_val) else card2_val
        cdef str suited = 's' if card1_str[1] == card2_str[1] else 'o'
        
        return high_card + low_card + suited

    cpdef setup_preflop(self, object hand=None):
        self.reset()
        self.assign_positions()
        self.handle_blinds()
        self.deal_private_cards(hand)
        self.num_actions = 0
        self.round_active_players = self.active_players()

    cpdef setup_postflop(self, str round_name):
        if self.is_terminal_river():
            return

        self.cur_round_index += 1
        if round_name == "flop":
            for _ in range(3):
                self.draw_card()
        else:
            self.draw_card()
        self.current_bet = 0
        for player in self.players:
            player.contributed_to_pot = 0
        self.round_active_players = self.active_players()
        self.player_index = (self.dealer_position + 1) % len(self.players)
        self.last_raiser = -1
        self.num_actions = 0

    cpdef bint handle_action(self, object action=None):
        if self.is_terminal() or self.is_terminal_river():
            return True

        if self.players[self.player_index].folded or self.players[self.player_index].chips == 0:
            self.player_index = (self.player_index + 1) % len(self.players)
            return self.is_terminal()

        if action:
            if self.players[self.player_index].take_action(self, self.player_index, action):
                self.last_raiser = self.player_index
        else:
            if self.players[self.player_index].get_action(self, self.player_index):
                self.last_raiser = self.player_index

        self.num_actions += 1
        self.player_index = (self.player_index + 1) % len(self.players)

        if self.active_players() == self.allin_players() + self.folded_players():
            self.progress_to_showdown()
            return True

        return self.is_terminal()

    cpdef progress_to_showdown(self):
        while self.num_board_cards() < 5:
            self.draw_card()
        self.cur_round_index = 4
        self.showdown()

    cpdef showdown(self):
        cdef unsigned long long player_hand
        cdef int best_score, player_score
        cdef unsigned long long zero = 0
        cdef int remaining_players = 0
        cdef list hands = [0] * len(self.players)  # Preallocate list

        for player in self.players:
            if not player.folded:
                remaining_players += 1

        if self.winner_index != -1:
            return  # Avoid redundant processing

        if remaining_players == 1:
            for i, player in enumerate(self.players):
                if not player.folded:
                    self.winner_index = i
                    break
        else:
            board_dupe = self.board
            deck_dupe = self.deck[:]
            for i, player in enumerate(self.players):
                hands[i] = player.hand

            win_rate = [0] * len(self.players)
            for _ in range(self.num_simulations):
                self.board = board_dupe
                self.deck = deck_dupe[:]
                self.fisher_yates_shuffle()

                while self.num_board_cards() < 5:
                    self.draw_card()

                best_score = -1
                self.winner_index = -1

                for i, player in enumerate(self.players):
                    if player.folded:
                        continue

                    if player.hand == zero:
                        player.hand = self.deck.pop() | self.deck.pop()

                    player_hand = player.hand | self.board
                    player_score = cy_evaluate(player_hand, 7)

                    if player_score > best_score:
                        best_score = player_score
                        self.winner_index = i
                
                for i in range(len(self.players)):
                    self.players[i].hand = hands[i]  # Restore hands

                win_rate[self.winner_index] += 1

            self.winner_index = win_rate.index(max(win_rate))
            num_simulations = float(self.num_simulations)  # Cache division factor

            for i, p in enumerate(self.players):
                p.expected_hand_strength = win_rate[i] / num_simulations

        winner = self.players[self.winner_index]
        winner.prior_gains += self.pot
        winner.chips += self.pot

        self.log_current_hand(terminal=True)  # Log the hand here if terminal at river


    cpdef bint is_terminal(self):
        if ((self.num_actions >= self.round_active_players and (self.last_raiser == -1 or self.last_raiser == self.player_index)) or
            self.active_players() == 1 or self.allin_players() == self.active_players()):
            return True
        return False

    cpdef bint is_terminal_river(self):
        if (self.cur_round_index >= 4 or
            (self.board_has_five_cards() and self.is_terminal()) or
            self.active_players() == 1 or self.allin_players() == self.active_players()):
            return True
        return False

    cdef void fisher_yates_shuffle(self):
        cdef int i, j
        cdef unsigned long long temp
        for i in range(len(self.deck) - 1, 0, -1):
            j = (numpy.random.rand() * 380204032 // 1) % (i + 1)
            temp = self.deck[i]
            self.deck[i] = self.deck[j]
            self.deck[j] = temp

    cpdef int allin_players(self):
        cdef int allin = 0
        for player in self.players:
            if player.chips == 0 and not player.folded:
                allin += 1
        return allin

    cpdef int active_players(self):
        cdef int alive = 0
        for player in self.players:
            if not player.folded:
                alive += 1
        return alive

    cpdef int folded_players(self):
        cdef int folded = 0
        for player in self.players:
            if player.folded:
                folded += 1
        return folded

    cpdef draw_card(self):
        self.board |= self.deck.pop()

    cpdef bint board_has_five_cards(self):
        return self.num_board_cards() == 5

    cpdef int num_board_cards(self):
        return bin(self.board).count('1')

    cpdef clone(self):
        # Clone players first
        new_players = []
        for i in range(len(self.players)):
            new_players.append(self.players[i].clone())

        # Create a new GameState instance
        cdef GameState new_state = GameState(new_players, self.small_blind, self.big_blind, self.num_simulations, self.silent, self.suits, self.values)

        # Copy all relevant attributes
        new_state.cur_round_index = self.cur_round_index
        new_state.dealer_position = self.dealer_position
        new_state.player_index = self.player_index
        new_state.num_actions = self.num_actions
        new_state.last_raiser = self.last_raiser
        new_state.pot = self.pot
        new_state.current_bet = self.current_bet
        new_state.board = self.board
        new_state.deck = self.deck[:]
        new_state.betting_history = [sublist[:] for sublist in self.betting_history]
        new_state.round_active_players = self.round_active_players
        new_state.winner_index = self.winner_index
        new_state.hand_id = self.hand_id

        return new_state

    cpdef debug_output(self):
        print(f"is_terminal(): {self.is_terminal()}")
        print(f"is_terminal_river(): {self.is_terminal_river()}")
        print(f"cur_round_index: {self.cur_round_index}")
        print(f"cur_pot: {self.pot}")
        print(f"cur_bet: {self.current_bet}")
        print(f"num_actions: {self.num_actions}")
        print(f"last_raiser: {self.last_raiser}")
        print(f"player_index: {self.player_index}")
        print(f"dealer_position: {self.dealer_position}")
        print(f"round_active_players: {self.round_active_players}")
        print(f"active_players(): {self.active_players()}")
        print(f"allin_players(): {self.allin_players()}")
        print(f"folded_players(): {self.folded_players()}")
        print(f"current_bet: {self.current_bet}")
        print(f"betting_history: {self.betting_history}")
        print(f"BOARD: {format_hand(self.board)}")
        print(f"deck: {[int_to_card(card) for card in self.deck]}")
        print(f"___________")
        for player in self.players:
            print(f"Player {player.player_index}")
            print(f"Player {player.player_index} position: {player.position}")
            print(f"Player {player.player_index} chips: {player.chips}")
            print(f"Player {player.player_index} folded: {player.folded}")
            print(f"Player {player.player_index} abstracted: {player.abstracted_hand}")
            print(f"Player {player.player_index} numeric hand: {format_hand(player.hand)}")
            print(f"Player {player.player_index} contributed: {player.contributed_to_pot}")
            print(f"Player {player.player_index} total contributed: {player.tot_contributed_to_pot}")
            print("")

    cpdef list generate_positions(self, int num_players):
        if num_players == 2:
            return ['D', 'SB']
        elif num_players == 3:
            return ['D', 'SB', 'BB']
        else:
            positions = ['D', 'SB', 'BB', 'UTG']
            if num_players > 4:
                positions.append('MP')
            if num_players > 5:
                positions.append('CO')
            if num_players == 7:
                positions.append('MP2')
            elif num_players == 8:
                positions.append('HJ')
            elif num_players == 9:
                positions.append('HJ')
                positions.append('MP2')
            return positions

    cpdef list create_deck(self, list suits, list values):
        cdef list deck = [card_to_int(suit, value) for suit in suits for value in values]
        return deck

cpdef unsigned long long card_to_int(str suit, str value):
    cdef unsigned long long one = 1
    cdef int suit_index = SUITS_INDEX[suit]
    cdef int value_index = VALUES_INDEX[value]
    cdef int bit_position = suit_index * 13 + value_index
    return one << bit_position

cpdef public list create_deck():
    cdef list deck = [card_to_int(suit, value) for suit in SUITS for value in VALUES]
    return deck

cpdef str int_to_card(unsigned long long card):
    cdef int bit_position = -1
    while card > 0:
        card >>= 1
        bit_position += 1
    cdef int suit_index = bit_position // 13
    cdef int value_index = bit_position % 13
    return f'{VALUES[value_index]}{SUITS[suit_index]}'

cpdef unsigned long long card_str_to_int(str card_str):
    return card_to_int(card_str[1], card_str[0])

cpdef list hand_to_cards(unsigned long long hand):
    cdef list cards = [card for card in create_deck() if card & hand]
    return cards

cpdef str format_hand(unsigned long long hand):
    cdef list cards = [int_to_card(card) for card in create_deck() if card & hand]
    return " ".join(cards)

cpdef display_game_state(GameState game_state, int player_index):
    print(f"______________________________________________________________________________")
    print(f"({game_state.players[player_index].position})Player {player_index + 1}: {format_hand(game_state.players[player_index].hand)}")
    print(f"Board: {format_hand(game_state.board)}")
    print(f"Pot: {game_state.pot}")
    print(f"Chips: {game_state.players[player_index].chips}")

    # print("______GAMESTATE______")
    # print(f"Active Players: {game_state.active_players()}")
    # print(f"Current Bet: {game_state.current_bet}")
    # print(f"Last Raiser: {game_state.last_raiser}")
    # print(f"Player Index: {game_state.player_index}")
    # print(f"Player Index Hand: {format_hand(game_state.players[game_state.player_index].hand)}")



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